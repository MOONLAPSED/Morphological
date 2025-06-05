#!/usr/bin/env python3
"""
Obsidian Publish Markdown and Attachment Crawler
Crawls an Obsidian Publish site using its cache to discover and download
all raw markdown files and attachments.
Uses only Python standard library (tested with 3.8+, suitable for 3.13).
"""

import urllib.request
import urllib.parse
import re
import os
import time
import gzip
import io
from pathlib import Path
from typing import Set, List, Dict, Tuple
import json

class ObsidianPublishDownloader:
    def __init__(self, site_url: str, site_id: str, output_dir: str = "obsidian_vault"):
        self.site_url = site_url.rstrip('/')
        self.site_id = site_id
        self.output_dir = Path(output_dir)
        self.raw_base_url = f"https://publish-01.obsidian.md/access/{site_id}/"
        
        self.failed_md_pages: Set[str] = set()
        self.failed_attachments: Set[str] = set()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _fetch_url(self, url: str, headers: Dict[str, str]) -> bytes:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
                if response.info().get('Content-Encoding') == 'gzip' or \
                   (content and content.startswith(b'\x1f\x8b')): # check content only if not None
                    content = gzip.decompress(content)
                return content
        except urllib.error.HTTPError as e:
            # Read the body of HTTP errors as it might contain useful info (like "Not Found" pages)
            error_content = b""
            try:
                error_content = e.read()
                if e.info().get('Content-Encoding') == 'gzip' or \
                   (error_content and error_content.startswith(b'\x1f\x8b')):
                   error_content = gzip.decompress(error_content)
            except Exception as e_read:
                print(f"Note: Could not read error body for {url}: {e_read}")
            
            # For "Not Found" type errors, we often get an HTML page or simple text.
            # Let the calling function decide if this content is an "actual" file or an error page.
            # For 404s specifically, it's likely an error page we don't want to save as content.
            if e.code == 404:
                 print(f"HTTP 404 Not Found for {url}. Server message (if any): {error_content[:200].decode('utf-8', 'ignore')}")
                 return b"" # Treat 404 as definitively not found

            print(f"Error fetching {url}: {e}. Body: {error_content[:200].decode('utf-8', 'ignore')}")
            return error_content # Return error body for other HTTP errors, might be useful
        except Exception as e:
            print(f"General error fetching {url}: {e}")
            return b""

    def fetch_site_cache(self) -> dict:
        cache_url = f"https://publish-01.obsidian.md/cache/{self.site_id}"
        print(f"Fetching site cache from: {cache_url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json,*/*',
            'Accept-Encoding': 'gzip, deflate'
        }
        content_bytes = self._fetch_url(cache_url, headers)
        if not content_bytes: return {}
        try:
            return json.loads(content_bytes.decode('utf-8'))
        except json.JSONDecodeError as e:
            print(f"Error decoding cache JSON: {e}")
            raw_cache_path = self.output_dir / "debug_cache_raw_error.txt"
            with open(raw_cache_path, 'wb') as f_raw: f_raw.write(content_bytes)
            print(f"Saved raw problematic cache content to {raw_cache_path}")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred processing cache data: {e}")
            return {}

    def _extract_paths_from_tree_recursive(self, tree_data) -> Set[str]:
        paths: Set[str] = set()
        if isinstance(tree_data, dict):
            if 'path' in tree_data and isinstance(tree_data['path'], str):
                paths.add(tree_data['path'])
            for key, value in tree_data.items():
                if key == 'path': continue
                if isinstance(value, (dict, list)):
                    paths.update(self._extract_paths_from_tree_recursive(value))
        elif isinstance(tree_data, list):
            for item in tree_data:
                paths.update(self._extract_paths_from_tree_recursive(item))
        return paths

    def extract_all_file_paths_from_cache(self, cache_data: dict) -> List[str]:
        all_file_paths: Set[str] = set()
        if not isinstance(cache_data, dict):
            print("Error: Cache data is not a dictionary.")
            return []

        is_likely_flat_dict = True
        if any(k in cache_data for k in ['files', 'tree', 'pages', 'config', 'customJS', 'customCSS', 'fileMap']):
             if isinstance(cache_data.get('files'), list) or \
                isinstance(cache_data.get('tree'), dict) or \
                isinstance(cache_data.get('fileMap'), dict):
                is_likely_flat_dict = False
        
        # Strategy 0: 'fileMap' key (common in some Obsidian Publish cache versions)
        if 'fileMap' in cache_data and isinstance(cache_data['fileMap'], dict):
            print("DEBUG: Using 'fileMap' strategy.")
            for path_with_extension in cache_data['fileMap'].keys():
                if isinstance(path_with_extension, str) and '.' in Path(path_with_extension).name:
                    all_file_paths.add(path_with_extension)
            if all_file_paths:
                print(f"DEBUG: Extracted {len(all_file_paths)} paths using 'fileMap'.")


        # Strategy 1: Cache data is a flat dictionary of {path: details/null}
        if not all_file_paths and is_likely_flat_dict:
            print("DEBUG: Using flat dictionary (cache_data.keys()) strategy.")
            for path_with_extension in cache_data.keys():
                if isinstance(path_with_extension, str) and '.' in Path(path_with_extension).name:
                    # Further filter common non-file keys if they somehow pass the 'is_likely_flat_dict'
                    if path_with_extension not in ['config', 'customJS', 'customCSS', 'version', 'notes', 'css', 'js']:
                        all_file_paths.add(path_with_extension)
            if all_file_paths:
                 print(f"DEBUG: Extracted {len(all_file_paths)} paths using flat dictionary strategy.")

        if not all_file_paths and 'files' in cache_data and isinstance(cache_data['files'], list):
            print("DEBUG: Trying 'files' list strategy.")
            for file_path_with_ext in cache_data['files']:
                if isinstance(file_path_with_ext, str): all_file_paths.add(file_path_with_ext)
            if all_file_paths: print(f"DEBUG: Extracted {len(all_file_paths)} paths using 'files' list strategy.")

        if not all_file_paths and 'tree' in cache_data and isinstance(cache_data['tree'], (dict, list)):
            print("DEBUG: Trying 'tree' structure strategy.")
            tree_paths = self._extract_paths_from_tree_recursive(cache_data['tree'])
            all_file_paths.update(tree_paths)
            if all_file_paths: print(f"DEBUG: Extracted {len(all_file_paths)} paths using 'tree' structure strategy.")
        
        if not all_file_paths: print("Warning: No file paths found in cache data using any known strategy.")
        return sorted(list(all_file_paths))

    def download_markdown_file_by_path(self, page_path_no_ext: str) -> bool:
        raw_url = "URL not constructed"
        try:
            path_for_url = page_path_no_ext.replace('\\', '/')
            encoded_path_components = urllib.parse.quote(path_for_url, safe='/')
            raw_url = f"{self.raw_base_url}{encoded_path_components}.md" 
            
            # Reduced verbosity for this print, progress shown in main loop
            # print(f"Downloading MD: {page_path_no_ext}.md from {raw_url}") 
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/plain,text/markdown,*/*;q=0.9',
                'Accept-Encoding': 'gzip, deflate'
            }
            md_content_bytes = self._fetch_url(raw_url, headers)

            if not md_content_bytes: # _fetch_url now returns b"" on 404 or other critical fetch errors
                print(f"Warning: No content retrieved for MD {page_path_no_ext}.md (URL: {raw_url}). Already logged by _fetch_url.")
                return False 
            
            md_content = md_content_bytes.decode('utf-8')

            # Check for Obsidian Publish's specific "File ... does not exist." message in content
            # This can happen if the URL is valid but points to a non-existent item within their system,
            # resulting in a 200 OK with an error message in the body.
            normalized_server_path_check = path_for_url # use the forward-slashed version
            if "## Not Found" in md_content and (
                f"File {normalized_server_path_check}.md does not exist" in md_content or
                f"File an equivalent of {normalized_server_path_check}.md could not be found" in md_content or # another common variant
                f"File an equivalent of {urllib.parse.unquote(encoded_path_components)}.md could not be found" in md_content # check unquoted too
            ):
                print(f"Error: Server returned 'Not Found' content for MD: {page_path_no_ext}.md. URL: {raw_url}")
                return False

            local_path = self.output_dir / (page_path_no_ext + '.md')
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(local_path, 'w', encoding='utf-8') as f: f.write(md_content)
            return True
        except UnicodeDecodeError as e:
            print(f"Error decoding MD content for {page_path_no_ext}.md: {e}. URL: {raw_url}")
            # Optionally save raw bytes for inspection
            error_bytes_path = self.output_dir / (page_path_no_ext + ".undecodable.md_bytes")
            error_bytes_path.parent.mkdir(parents=True, exist_ok=True)
            with open(error_bytes_path, "wb") as f_err_bytes: f_err_bytes.write(md_content_bytes)
            print(f"Saved undecodable raw bytes to {error_bytes_path}")
            return False
        except Exception as e:
            print(f"Error processing/saving MD {page_path_no_ext}.md: {e}. URL attempted: {raw_url}")
            return False

    def download_attachment_by_path(self, attachment_path_with_ext: str) -> bool:
        raw_url = "URL not constructed"
        try:
            path_for_url = attachment_path_with_ext.replace('\\', '/')
            encoded_path_components = urllib.parse.quote(path_for_url, safe='/')
            raw_url = f"{self.raw_base_url}{encoded_path_components}"
            
            # print(f"Downloading Attachment: {attachment_path_with_ext} from {raw_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': '*/*', 
                'Accept-Encoding': 'gzip, deflate'
            }
            attachment_content_bytes = self._fetch_url(raw_url, headers)

            if not attachment_content_bytes:
                print(f"Warning: No content retrieved for attachment {attachment_path_with_ext} (URL: {raw_url}). Already logged by _fetch_url.")
                return False 

            local_path = self.output_dir / attachment_path_with_ext
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(local_path, 'wb') as f: f.write(attachment_content_bytes)
            return True
        except Exception as e:
            print(f"Error downloading attachment {attachment_path_with_ext}: {e}. URL attempted: {raw_url}")
            return False

    def run_download(self):
        print(f"Starting download process for site ID: {self.site_id}")
        print(f"Output directory: {self.output_dir}")
        
        cache_data = self.fetch_site_cache()
        if not cache_data:
            print("Failed to fetch or parse site cache. Aborting.")
            return

        cache_debug_path = self.output_dir / "debug_site_cache.json"
        try:
            with open(cache_debug_path, 'w', encoding='utf-8') as f: json.dump(cache_data, f, indent=2)
            print(f"DEBUG: Complete site cache saved to {cache_debug_path}")
        except Exception as e: print(f"DEBUG: Error saving site cache json: {e}")

        all_file_paths = self.extract_all_file_paths_from_cache(cache_data)
        if not all_file_paths:
            print("No file paths extracted from cache data. Nothing to download.")
            return

        md_paths_to_download_no_ext: List[str] = []
        attachment_paths_to_download_with_ext: List[str] = []

        for path_with_ext in all_file_paths:
            if not isinstance(path_with_ext, str) or not path_with_ext.strip():
                print(f"Warning: Invalid empty or non-string file path found in cache: '{path_with_ext}'")
                continue
            
            path_obj_original_slashes = Path(path_with_ext) # Keep original slashes for now
            # Use original slashes from cache for list, then normalize for URL
            
            if path_obj_original_slashes.suffix.lower() == '.md':
                # Store with original slashes (typically '/' from cache)
                md_paths_to_download_no_ext.append(str(path_obj_original_slashes.with_suffix('')))
            else:
                if path_obj_original_slashes.name:
                    attachment_paths_to_download_with_ext.append(str(path_obj_original_slashes))
        
        md_paths_to_download_no_ext = sorted(list(set(md_paths_to_download_no_ext)))
        attachment_paths_to_download_with_ext = sorted(list(set(attachment_paths_to_download_with_ext)))

        print(f"Identified {len(md_paths_to_download_no_ext)} markdown files and {len(attachment_paths_to_download_with_ext)} attachments.")
        successful_md, successful_attachments = 0, 0
        total_items = len(md_paths_to_download_no_ext) + len(attachment_paths_to_download_with_ext)
        processed_items = 0
        delay_between_requests = 0.05

        print("\n--- Downloading Markdown Files ---")
        for md_path_no_ext in md_paths_to_download_no_ext:
            processed_items += 1
            # Construct display path with OS-specific separators for print statement
            display_path = str(Path(md_path_no_ext + ".md"))
            print(f"Progress: {processed_items}/{total_items} (MD: {display_path})")
            if self.download_markdown_file_by_path(md_path_no_ext):
                successful_md += 1
            else:
                self.failed_md_pages.add(md_path_no_ext + ".md")
            time.sleep(delay_between_requests)

        print("\n--- Downloading Attachments ---")
        for att_path_with_ext in attachment_paths_to_download_with_ext:
            processed_items += 1
            display_path = str(Path(att_path_with_ext))
            print(f"Progress: {processed_items}/{total_items} (Attachment: {display_path})")
            if self.download_attachment_by_path(att_path_with_ext):
                successful_attachments += 1
            else:
                self.failed_attachments.add(att_path_with_ext)
            time.sleep(delay_between_requests)
        
        print("\n--- Download Summary ---")
        print(f"Markdown Files: {successful_md} downloaded, {len(self.failed_md_pages)} failed.")
        print(f"Attachments: {successful_attachments} downloaded, {len(self.failed_attachments)} failed.")
        
        if self.failed_md_pages:
            print("\nFailed Markdown Pages:")
            for page_path in sorted(self.failed_md_pages): print(f"  - {str(Path(page_path))}")
        if self.failed_attachments:
            print("\nFailed Attachments:")
            for att_path in sorted(self.failed_attachments): print(f"  - {str(Path(att_path))}")
        
        summary_data = {
            'site_id': self.site_id, 'output_directory': str(self.output_dir),
            'total_markdown_files_identified': len(md_paths_to_download_no_ext),
            'markdown_files_downloaded': successful_md, 'markdown_files_failed': len(self.failed_md_pages),
            'failed_markdown_files_list': sorted([str(Path(p)) for p in self.failed_md_pages]),
            'total_attachments_identified': len(attachment_paths_to_download_with_ext),
            'attachments_downloaded': successful_attachments, 'attachments_failed': len(self.failed_attachments),
            'failed_attachments_list': sorted([str(Path(p)) for p in self.failed_attachments]),
        }
        summary_path = self.output_dir / 'download_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f: json.dump(summary_data, f, indent=2)
        print(f"\nDownload summary saved to: {summary_path}")

def main():
    SITE_URL = "https://publish.obsidian.md/pkc"
    SITE_ID = "ca2bc62eb522ee33e3699f9e9e4a8e91"
    OUTPUT_DIR = "pkc_vault_downloaded"
    
    downloader = ObsidianPublishDownloader(SITE_URL, SITE_ID, OUTPUT_DIR)
    downloader.run_download()

if __name__ == "__main__":
    main()