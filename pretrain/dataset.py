import typing
import random
import time
import requests
import boto3
import smart_open
import numpy as np
import bittensor as bt
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset
from pprint import pprint

import os
from dotenv import load_dotenv
load_dotenv()

import concurrent.futures

class SubsetLoader(IterableDataset):
    """Base class for data-specific subset loader classes."""

    name: str = None  # Dataset name
    rows_base_url: str = "https://hf.lester.my/rows"
    size_base_url: str = "https://hf.lester.my/size"
    max_pages: int = None

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = False,
        random_seed: typing.Optional[int] = None,
        config: str = "default",
        split: str = "train",
        requires_auth: bool = False,
        num_threads: int = 16,  # Added num_threads parameter
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_pages = num_pages
        self.tokenizer = tokenizer
        self.pack_samples = pack_samples
        self.config = config
        self.split = split
        self.requires_auth = requires_auth
        self.num_threads = num_threads

        # Initialize with seed if provided
        if random_seed is not None:
            random.seed(random_seed)

        self.num_rows_per_page = 50
        self.duplicate_page_threshold = 100
        self.retry_limit = 10
        self.retry_delay = 5

        # Buffers
        self.buffer = []
        self.used_buffer = []
        self.padded_buffer = []
        self.session = requests.Session()
        # Get HF token if needed
        self.hf_token = None
        if self.requires_auth:
            self.hf_token = os.getenv("HF_TOKEN")
            if not self.hf_token:
                raise ValueError("HF_TOKEN environment variable not found")

        # Initialize request params
        self.params = self._get_default_params()

        # Fetch pages if specified
        # If the fetched pages are empty, try again until
        # we hit the retry limit.
        fetch_attempt = 1

        if self.num_pages:
            while fetch_attempt < self.retry_limit:
                self._initialize_pages()
                fetch_attempt += 1

                # Exit if the buffer has at least one batch
                if len(self.buffer) >= self.sequence_length:
                    break

                bt.logging.warning(
                    f"All fetched pages seem to be empty or have an extremely low token count. "
                    f"Trying to fetch a new set of pages... (attempt {fetch_attempt}/{self.retry_limit})"
                )

            # If we exhaust all attempts and still don't have enough data, raise an error
            if len(self.buffer) < self.sequence_length:
                raise ValueError(
                    "Maximum retry limit for fetching pages reached. "
                    "All fetched pages seem to be empty or have an extremely low token count."
                )

    def _get_default_params(self):
        """Get default request parameters. Override if needed."""
        return {
            "dataset": self.name,
            "config": self.config,
            "split": self.split,
        }

    def _get_request_headers(self):
        """Get request headers. Override if needed."""
        headers = {}
        if self.requires_auth:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        return headers

    def _initialize_pages(self):
        """Initialize pages based on loader type"""
        if hasattr(self, 'fetch_dataset_configs'):
            # For FineWebEdu2 style loaders
            self.configs_data = self.fetch_dataset_configs()
            self._fetch_data_to_buffer(self.num_pages)
        else:
            # For simple page-based loaders
            pages = self._sample_pages()
            self.fetch_data_for_pages(pages)

    def fetch_data_for_pages(self, pages):
        """Set the pages and fetch their data to the buffer."""
        self.pages = pages
        self.buffer = []
        for page in self.pages:
            self._fetch_data_for_page(page)

    def _fetch_data_for_page(self, page):
        """Fetch data for a single page"""
        # Handle different page types (tuple vs int)
        if isinstance(page, tuple):
            config_name, page_num, split = page
            self.params.update({
                "config": config_name,
                "split": split,
                "offset": page_num,
            })
        else:
            self.params["offset"] = page

        self.params["length"] = self.num_rows_per_page

        attempt = 0
        while attempt < self.retry_limit:
            try:
                response = self.session.get(
                    self.rows_base_url,
                    params=self.params,
                    headers=self._get_request_headers()
                )
                bt.logging.success(
                    f" response {response}"
                )
                response.raise_for_status()

                for row in response.json()["rows"]:
                    content = self._get_content_from_row(row)
                    input_ids = self.tokenizer(content, truncation=True)["input_ids"]
                    self.buffer += input_ids
                    self.buffer += [self.tokenizer.eos_token_id]

                break

            except requests.exceptions.RequestException as e:
                attempt += 1
                bt.logging.warning(
                    f"Failed to fetch data for page {page}, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)
                else:
                    bt.logging.error("Maximum retry limit reached. Unable to fetch data.")
                    raise

    def _get_content_from_row(self, row):
        """Extract content from row based on dataset format. Override if needed."""
        return row["row"].get("text", row["row"].get("content"))

    def _sample_pages(self):
        """Sample random pages. Override for custom sampling logic."""
        return [random.randint(1, self.max_pages) for _ in range(self.num_pages)]

    def get_page_names(self):
        """Get page names in consistent format"""
        if not hasattr(self, 'pages'):
            return []

        if isinstance(self.pages[0], tuple):
            return [f"{cfg_name}_{num_rows}_{split}"
                   for cfg_name, num_rows, split in self.pages]
        return self.pages

    def _get_pad_size(self, input_ids):
        """Get padding size for input tokens."""
        if self.pack_samples:
            return 1

        sample_size = len(input_ids)
        remainder = sample_size % self.sequence_length
        pad_size = self.sequence_length - remainder
        pad_size = pad_size % self.sequence_length
        return pad_size

    def _refill_padded_buffer(self):
        """Refill the padded buffer from the main buffer."""
        while self.buffer and len(self.padded_buffer) < self.sequence_length:
            input_ids = []
            EOS_index = self.buffer.index(self.tokenizer.eos_token_id)
            input_ids = self.buffer[: EOS_index + 1]
            self.buffer = self.buffer[EOS_index + 1 :]
            self.used_buffer += input_ids
            self.padded_buffer += input_ids[:-1]
            self.padded_buffer += [self.tokenizer.eos_token_id] * self._get_pad_size(
                input_ids=input_ids[:-1]
            )

    def __iter__(self):
        self.buffer = self.used_buffer + self.buffer
        self.padded_buffer = []
        self._refill_padded_buffer()
        return self

    def __next__(self):
        batch = []
        while len(self.padded_buffer) >= self.sequence_length:
            batch.append(self.padded_buffer[: self.sequence_length])
            self.padded_buffer = self.padded_buffer[self.sequence_length :]
            self._refill_padded_buffer()
            if len(batch) == self.batch_size:
                return np.stack(batch)
        raise StopIteration


class SubsetPes2oXLoader(SubsetLoader):
    max_pages: int = 8242000
    name: str = "laion/Pes2oX-fulltext"

    def __init__(self, **kwargs):
        super().__init__(config="pes2ov2", **kwargs)


class SubsetStackV1DedupLoader(SubsetLoader):
    max_pages: int = 236655813
    name: str = "bigcode/the-stack-dedup"

    def __init__(self, **kwargs):
        super().__init__(requires_auth=True, **kwargs)

class SubsetStackV2DedupLoader(SubsetLoader):
    max_pages: int = 5_451_114_734
    name: str = "bigcode/the-stack-v2-dedup"

    def __init__(self, **kwargs):

        # Create an AWS S3 session to enable reading data
        session = boto3.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])

        self.s3_sess = session.client("s3")

        super().__init__(requires_auth=True, **kwargs)


    def _download_row_content(self, blob_id, src_encoding):
        """Download the row content from S3.
        """

        s3_url = f"https://softwareheritage.s3.amazonaws.com/content/{blob_id}"

        with smart_open.open(s3_url, "rb", compression=".gz", transport_params={"client": self.s3_sess}) as fin:
            content = fin.read().decode(src_encoding)

        return content

    def _get_content_from_row(self, row):
        """Extract row content by downloading from S3 """

        content = self._download_row_content(row['row']['blob_id'], row['row']['src_encoding'])
        return content


class SubsetFalconLoader(SubsetLoader):
    max_pages: int = 968000015
    name: str = "tiiuae/falcon-refinedweb"


class SubsetFineWebEdu2Loader(SubsetLoader):
    name: str = "HuggingFaceFW/fineweb-edu-score-2"    
    
    def __init__(self, **kwargs):
        self.session = requests.Session()
        super().__init__(**kwargs)
    
    def fetch_dataset_configs(self) -> typing.Dict[str, typing.Dict]:
        """
        Fetch dataset configs and their metadata.
        Returns a dictionary with config names as keys and metadata as values.
        """
        params = dict(dataset=self.name)

        attempt = 0
        while attempt < self.retry_limit:
            try:
                response = self.session.get(self.size_base_url, params=params)
                response.raise_for_status()

                configs_dict = response.json()["size"]["splits"]
                configs_data = {
                    entry["config"]: {
                        "num_rows": entry["num_rows"],
                        "split": entry["split"],
                    }
                    for entry in configs_dict
                    if entry["config"] != "default"
                }

                return configs_data

            except requests.exceptions.RequestException as e:
                attempt += 1
                bt.logging.warning(
                    f"Failed to fetch dataset configs, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)
                else:
                    bt.logging.error("Maximum retry limit reached. Unable to fetch data.")
                    raise

    def _fetch_data_for_page_threaded(self, page):
      """Fetches data for a single page, to be used in a thread."""
      # Handle different page types (tuple vs int)
      if isinstance(page, tuple):
          config_name, page_num, split = page
          self.params.update({
              "config": config_name,
              "split": split,
              "offset": page_num,
          })
      else:
          self.params["offset"] = page

      self.params["length"] = self.num_rows_per_page
      attempt = 0
      while attempt < self.retry_limit:
          try:
              response = self.session.get(
                  self.rows_base_url,
                  params=self.params,
                  headers=self._get_request_headers()
              )
              response.raise_for_status()
              input_ids_for_page = []
              for row in response.json()["rows"]:
                  content = row["row"]["text"]
                  input_ids = self.tokenizer(content, truncation=True)["input_ids"]
                  input_ids_for_page += input_ids
                  input_ids_for_page += [self.tokenizer.eos_token_id]
              return input_ids_for_page

          except requests.exceptions.RequestException as e:
              attempt += 1
              bt.logging.warning(
                  f"Failed to fetch data for page {page}, retrying. Attempt {attempt}/{self.retry_limit}"
              )
              if attempt < self.retry_limit:
                  time.sleep(self.retry_delay)
              else:
                  bt.logging.error("Maximum retry limit reached. Unable to fetch data.")
                  raise
      return []

    def _fetch_data_to_buffer(self, num_pages):
        """Fetch data to buffer with support for multiple configs using threads."""
        self.pages = []
        attempts = 0
        duplicates = 0
        initial_offset = random.randint(0, self.num_rows_per_page - 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_page = {}
            while len(self.pages) < num_pages:
                page = self.get_random_pages(num_pages=1, initial_offset=initial_offset)[0]

                if page in self.pages:
                    duplicates += 1
                    if duplicates >= self.duplicate_page_threshold:
                        bt.logging.debug(
                            f"Hit duplicate page threshold of {self.duplicate_page_threshold}. "
                            f"Stopping early at: {len(self.pages)} pages."
                        )
                        break
                    continue
                
                future = executor.submit(self._fetch_data_for_page_threaded, page)
                future_to_page[future] = page
                self.pages.append(page)

            # Collect results as they become available
            for future in concurrent.futures.as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    input_ids_for_page = future.result()
                    self.buffer += input_ids_for_page
                except Exception as e:
                  bt.logging.warning(f"Error fetching data for page {page}: {e}")

    def get_random_pages(self, num_pages, initial_offset):
        """Get random pages across different configs."""
        pages = []
        for _ in range(num_pages):
            config_name = random.choice(list(self.configs_data.keys()))
            data_row_count = self.configs_data[config_name]["num_rows"] - initial_offset
            data_page_count = (data_row_count + 1) // self.num_rows_per_page
            selected_page_start = initial_offset + (
                random.randint(0, data_page_count - 1) * self.num_rows_per_page
            )
            split = self.configs_data[config_name]["split"]
            pages.append((config_name, selected_page_start, split))
        return pages


    def fetch_data_to_rows(self, num_pages):
      """Fetch data and return raw text rows instead of adding to buffer, using threads."""
      downloaded_pages = set()
      rows = []
      attempts = 0
      duplicates = 0
      initial_offset = random.randint(0, self.num_rows_per_page - 1)

      with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
          future_to_page = {}
          while len(downloaded_pages) < num_pages:
              page = self.get_random_pages(num_pages=1, initial_offset=initial_offset)[0]

              if page in downloaded_pages:
                  duplicates += 1
                  if duplicates >= self.duplicate_page_threshold:
                      bt.logging.debug(
                          f"Hit duplicate page threshold of {self.duplicate_page_threshold}. "
                          f"Stopping early at: {len(downloaded_pages)} pages."
                      )
                      break
                  continue
              
              future = executor.submit(self._fetch_data_for_page_threaded, page)
              future_to_page[future] = page
              downloaded_pages.add(page)
          for future in concurrent.futures.as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    input_ids_for_page = future.result()
                    for input_ids in input_ids_for_page:
                        decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                        rows.append(decoded_text)

                except Exception as e:
                    bt.logging.warning(f"Error fetching data for page {page}: {e}")
      return rows
