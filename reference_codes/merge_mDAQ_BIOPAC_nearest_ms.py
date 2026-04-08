import os
import csv
import logging
import psutil
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm
import bisect
import time
import statistics
from typing import Dict, List, Tuple, Optional

# Global variable to track overall start time for performance metrics
start_all = 0.0

# --- Utility Functions ---

def find_nearest_timestamp(target: int, timestamps: List[int], window_ms: int = 1000) -> Optional[int]:
    """
    Find the first timestamp in a sorted list that is an exact match
    or the next available timestamp within a specified window.

    Args:
        target: The target timestamp (in milliseconds).
        timestamps: A sorted list of timestamps (in milliseconds) to search within.
        window_ms: The maximum allowed difference (in milliseconds) between the target
                   and the next available timestamp. Defaults to 1000ms.

    Returns:
        The matched timestamp (int) if found within the window, otherwise None.
    """
    if not timestamps:
        return None
    idx = bisect.bisect_left(timestamps, target)
    if idx < len(timestamps) and timestamps[idx] - target <= window_ms:
        return timestamps[idx]
    return None

def setup_logger(log_dir: str, log_filename: str = 'processing.log') -> None:
    """
    Sets up logging to write messages to both the console and a file.
    Removes existing handlers before adding new ones.

    Args:
        log_dir: The directory to save the log file.
        log_filename: The name for the log file. Defaults to 'processing.log'.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Ensure logger level is set

    # Remove existing handlers to prevent duplicate logs or file locks
    for handler in logger.handlers[:]:
        try:
            handler.close() # Close the handler first
        except Exception as e:
            # Use print as logger might be mid-config or closing failed handler
            print(f"Debug: Error closing handler {handler}: {e}")
        logger.removeHandler(handler)

    # Create new handlers
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    stream_handler = logging.StreamHandler()

    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add new handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Log initialization confirmation *after* setup is complete
    # logging.info(f"Logger initialized. Log file: {log_filepath}") # Logged in main now


def select_folders() -> Tuple[str, str, str, str]:
    """
    Uses Tkinter dialogs to prompt the user to select the BIOPAC file,
    label file, mDAQ folder, and output folder.

    Returns:
        A tuple containing the paths:
        (biopac_file, mdaq_folder, label_file, output_folder)

    Raises:
        FileNotFoundError: If the BIOPAC or label file is not selected.
        NotADirectoryError: If the mDAQ or output folder is not selected.
    """
    root = tk.Tk()
    root.withdraw() # Hide the main Tkinter window

    messagebox.showinfo("Instructions", "Please select the required files and folders:\n1. BIOPAC .txt file\n2. Label .csv or .txt file\n3. Folder containing mDAQ .csv files\n4. Output folder for results")

    # Select BIOPAC file
    biopac_file = filedialog.askopenfilename(
        title="1. Select BIOPAC file (*.txt)",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not biopac_file:
        messagebox.showerror("Error", "BIOPAC file selection cancelled.")
        raise FileNotFoundError("BIOPAC file not selected.")
    print(f"BIOPAC file selected: {biopac_file}")

    # Select Label file
    label_file = filedialog.askopenfilename(
        title="2. Select Label file (*.csv, *.txt)",
        filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not label_file:
        messagebox.showerror("Error", "Label file selection cancelled.")
        raise FileNotFoundError("Label file not selected.")
    print(f"Label file selected: {label_file}")

    # Select mDAQ folder
    mdaq_folder = filedialog.askdirectory(
        title="3. Select mDAQ folder containing CSV files"
    )
    if not mdaq_folder:
        messagebox.showerror("Error", "mDAQ folder selection cancelled.")
        raise NotADirectoryError("mDAQ folder not selected.")
    print(f"mDAQ folder selected: {mdaq_folder}")

    # Select Output folder
    output_folder = filedialog.askdirectory(
        title="4. Select Output folder for processed data"
    )
    if not output_folder:
        messagebox.showerror("Error", "Output folder selection cancelled.")
        raise NotADirectoryError("Output folder not selected.")
    print(f"Output folder selected: {output_folder}")

    root.destroy() # Close the Tkinter root window
    return biopac_file, mdaq_folder, label_file, output_folder

# --- Data Processing Functions ---

def process_biopac(biopac_file: str) -> Tuple[int, int, Dict[int, List[str]], int]:
    """
    Processes a BIOPAC text file (.txt) exported from AcqKnowledge.
    Detects sample rate, start time, and dynamically maps channels based on
    Name/Unit pairs found before the data header, corresponding to CH columns.
    Extracts data specifically for ECG, PPG, SKT, and EDA channels.
    Calculates timestamps based on sample rate and row index to avoid collisions.

    Args:
        biopac_file: Path to the BIOPAC .txt file.

    Returns:
        A tuple containing:
        - start_time_ms (int): Recording start timestamp in milliseconds.
        - end_time_ms (int): Recording end timestamp in milliseconds.
        - biopac_data (dict): Dictionary mapping timestamps (int) to a list of
                                data values (str) in the fixed order [ECG, PPG, SKT, EDA].
                                Empty strings are used if a channel is not found/mapped.
        - processed_count (int): Number of data rows successfully processed and added to dict.

    Raises:
        ValueError: If essential header information (sample rate, start time, data header,
                    channel descriptions) cannot be found or parsed correctly.
        FileNotFoundError: If the biopac_file does not exist.
    """
    biopac_data: Dict[int, List[str]] = {}
    processed_count = 0 # Initialize counter
    logging.info(f'Starting BIOPAC data processing for: {biopac_file}')
    start_process_time = time.time()

    try:
        with open(biopac_file, 'r', encoding='utf-8') as f:
            lines = [line for line in f] # Read all lines into memory
    except FileNotFoundError:
        logging.error(f"BIOPAC file not found: {biopac_file}")
        raise
    except Exception as e:
        logging.error(f"Error reading BIOPAC file {biopac_file}: {e}")
        raise

    # --- Header Parsing ---
    sample_rate_ms: Optional[float] = None
    start_time_ms: Optional[int] = None
    data_header_line: Optional[str] = None
    data_header_idx: Optional[int] = None

    # Find key header lines and their indices
    for idx, line in enumerate(lines):
        line_strip = line.strip()
        if not line_strip: continue # Skip empty lines

        # Find Sample Rate
        if "msec/sample" in line_strip and sample_rate_ms is None:
            try:
                sample_rate_ms = float(line_strip.split()[0])
                logging.info(f"Detected sample rate: {sample_rate_ms} msec/sample at line {idx+1}")
            except (ValueError, IndexError):
                logging.warning(f"Could not parse sample rate from line {idx+1}: {line_strip}")

        # Find Recording Start Time
        elif line_strip.startswith('Recording on:') and start_time_ms is None:
            time_str = line_strip.split(': ', 1)[1]
            try:
                start_datetime = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                try:
                    start_datetime = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    logging.warning(f"Could not parse recording timestamp at line {idx+1}: {time_str}")
                    continue
            start_time_ms = int(start_datetime.timestamp() * 1000)
            logging.info(f"Detected recording start time: {start_datetime} ({start_time_ms} ms) at line {idx+1}")

        # Find Data Header Line (e.g., 'sec,CH1,CH2,...')
        elif line_strip.startswith('sec,CH') and data_header_line is None:
            data_header_line = line_strip
            data_header_idx = idx
            logging.info(f"Found data header at line {idx + 1}: {data_header_line}")
            # We can break here as channel descriptions should precede this line
            break

    # --- Validation of Essential Header Info ---
    if sample_rate_ms is None:
        raise ValueError("Could not find or parse sample rate (msec/sample) in file header.")
    if start_time_ms is None:
        raise ValueError("Could not find or parse 'Recording on:' timestamp in file header.")
    if data_header_line is None or data_header_idx is None:
        raise ValueError("Could not find data header line (starting with 'sec,CH') in file.")

    # --- Channel Mapping (Working Backwards from Data Header) ---
    headers = [h.strip() for h in data_header_line.split(',')]
    if not headers[-1]: headers.pop() # Remove empty string if header ends with comma
    ch_headers = headers[1:] # Get only the CHx headers (e.g., ['CH1', 'CH2', 'CH9'])
    num_data_channels = len(ch_headers)
    logging.info(f"Channel headers detected: {ch_headers} ({num_data_channels} channels)")

    if num_data_channels <= 0:
        raise ValueError(f"No data channels (CHx) found in header: {data_header_line}")

    # Create mapping from CHx header to its column index (1-based for data)
    header_indices = {header: i + 1 for i, header in enumerate(ch_headers)} # {'CH1': 1, 'CH2': 2, 'CH9': 5}

    # Define target channels and keywords
    target_channel_info = {
        'ECG': {'keywords': ['ECG']},
        'PPG': {'keywords': ['PPG']},
        'SKT': {'keywords': ['SKT', 'Temp', 'Temperature']},
        'EDA': {'keywords': ['EDA', 'GSR', 'SC', 'Siemens']}
    }

    # Map target channel names (ECG, PPG, etc.) to their column index in the data
    channel_positions: Dict[str, int] = {} # e.g., {'ECG': 4, 'PPG': 3, ...}
    mapped_ch_headers = set() # Keep track of which CHx headers we successfully mapped

    # Start reading pairs (Unit, Name) backwards from the line before the data header line
    current_line_idx = data_header_idx - 1 # Start at the line expected to be the last Unit

    logging.info(f"Starting channel mapping by reading backwards from line {current_line_idx + 1}")

    for i in range(num_data_channels - 1, -1, -1): # Iterate backwards through CH headers (e.g., CH9, CH4, ...)
        ch_header = ch_headers[i] # The CHx header we are trying to map (e.g., 'CH9')

        # Ensure we don't read past the beginning of the file
        if current_line_idx < 1:
            logging.error(f"Reached beginning of file while trying to read Name/Unit for {ch_header}. Check file structure.")
            break # Stop mapping if we run out of lines

        # Read Unit line and Name line
        unit_line = lines[current_line_idx].strip()
        name_line = lines[current_line_idx - 1].strip()
        logging.debug(f"Mapping {ch_header}: Reading Name='{name_line}' (line {current_line_idx}), Unit='{unit_line}' (line {current_line_idx+1})")

        # Check if this Name line matches any of our target channels' keywords
        found_match = False
        for target_name, info in target_channel_info.items():
            # Only map if this target channel hasn't been assigned yet
            if target_name not in channel_positions:
                name_line_lower = name_line.lower()
                for keyword in info['keywords']:
                    if keyword.lower() in name_line_lower:
                        column_index = header_indices[ch_header]
                        channel_positions[target_name] = column_index
                        mapped_ch_headers.add(ch_header)
                        logging.info(f"Mapped '{target_name}' to {ch_header} (Column Index: {column_index}) based on Name: '{name_line}'")
                        found_match = True
                        break # Keyword found for this target_name, move to next CH header
            if found_match:
                break # Target channel mapped, move to next CH header

        if not found_match:
             logging.warning(f"No target keyword match found for {ch_header} with Name: '{name_line}'")

        # Move up two lines for the next Name/Unit pair
        current_line_idx -= 2

    logging.info(f"Final channel mapping (Target Channel -> Column Index): {channel_positions}")

    # --- Validation of Mapping ---
    all_targets_found = True
    for target in target_channel_info:
        if target not in channel_positions:
            logging.warning(f"Target channel '{target}' was not found or mapped.")
            all_targets_found = False
    if all_targets_found:
        logging.info("All target channels (ECG, PPG, SKT, EDA) were successfully mapped.")


    # --- Data Processing ---
    # Data starts 2 lines after the data header line
    data_start_idx = data_header_idx + 2
    logging.info(f"Starting data processing from line {data_start_idx + 1}")

    skipped_count = 0
    last_timestamp_ms = start_time_ms # Initialize with start time
    collision_check: Dict[int, int] = {} # To check for timestamp collisions

    # Define the fixed order for output data columns
    output_order = ['ECG', 'PPG', 'SKT', 'EDA']

    for line_idx, line in enumerate(tqdm(lines[data_start_idx:], desc="Processing BIOPAC data", unit="row")):
        line_strip = line.strip()
        if not line_strip:
            skipped_count += 1
            continue # Skip empty lines

        values = [v.strip() for v in line_strip.split(',')]
        if not values[-1]: values.pop() # Remove empty string if line ends with comma

        if len(values) != len(headers):
             logging.warning(f"Skipping line {data_start_idx + line_idx + 1}: Incorrect number of columns ({len(values)}). Expected {len(headers)}. Line: '{line_strip}'")
             skipped_count += 1
             continue

        try:
            # --- NEW TIMESTAMP CALCULATION ---
            # Calculate timestamp based on row index (k) and sample rate
            # k is the 0-based index relative to the first data row
            k = line_idx
            time_offset_ms = k * sample_rate_ms # Use the detected float sample rate
            # Use round() to get the nearest integer millisecond
            timestamp_ms = start_time_ms + round(time_offset_ms)
            # --- END NEW TIMESTAMP CALCULATION ---

            # Check for collisions (optional, for debugging)
            if timestamp_ms in collision_check:
                collision_check[timestamp_ms] += 1
                # Log collision only once per timestamp to avoid flooding logs
                if collision_check[timestamp_ms] == 2:
                    logging.warning(f"Timestamp collision detected for {timestamp_ms} ms at line approx {data_start_idx + line_idx + 1}.")
            else:
                collision_check[timestamp_ms] = 1

            last_timestamp_ms = timestamp_ms # Keep track of the last valid timestamp

            # Extract data values based on the mapped positions, in the fixed output order
            extracted_values = []
            for target_ch in output_order:
                if target_ch in channel_positions:
                    col_idx = channel_positions[target_ch]
                    if 0 <= col_idx < len(values):
                         extracted_values.append(values[col_idx])
                    else:
                         logging.warning(f"Column index {col_idx} for '{target_ch}' out of range on line {data_start_idx + line_idx + 1}. Appending empty string.")
                         extracted_values.append('')
                else:
                    extracted_values.append('') # Append empty string if channel wasn't mapped

            biopac_data[timestamp_ms] = extracted_values
            # processed_count is incremented later based on final dict size

        except (ValueError, IndexError) as e:
            logging.warning(f"Skipping line {data_start_idx + line_idx + 1}: Error processing values or timestamp. Error: {e}. Line: '{line_strip}'")
            skipped_count += 1
            continue

    # Calculate final processed count based on the number of unique timestamps generated
    processed_count = len(biopac_data)
    num_collisions = sum(count - 1 for count in collision_check.values() if count > 1)

    processing_time = time.time() - start_process_time
    logging.info(f'Finished BIOPAC data processing.')
    logging.info(f'Processing time: {processing_time:.2f} seconds')
    logging.info(f'BIOPAC samples added to dictionary (unique timestamps): {processed_count}')
    logging.info(f'BIOPAC lines skipped: {skipped_count}')
    if num_collisions > 0:
        logging.warning(f'BIOPAC timestamp collisions detected (overwritten rows): {num_collisions}')
    else:
        logging.info('No BIOPAC timestamp collisions detected.')


    if not biopac_data:
        logging.warning("No BIOPAC data was successfully processed.")
        return start_time_ms, start_time_ms, biopac_data, 0 # Return 0 processed count

    # Determine end time from the last processed timestamp
    end_time_ms = last_timestamp_ms
    logging.info(f"BIOPAC data range: {start_time_ms} ms to {end_time_ms} ms")

    return start_time_ms, end_time_ms, biopac_data, processed_count


def process_mdaq(mdaq_folder: str, start_time_ms: int) -> Tuple[Dict[int, List[str]], int, int]:
    """
    Processes mDAQ data from a folder containing sequentially numbered CSV files.
    Calculates timestamps based on the initial BIOPAC start time and accumulated
    inter-sample intervals (ISI) from the mDAQ files.

    Args:
        mdaq_folder: Path to the folder containing mDAQ CSV files (e.g., 1.csv, 2.csv...).
        start_time_ms: The starting timestamp obtained from the BIOPAC file (milliseconds).

    Returns:
        A tuple containing:
        - mdaq_data (dict): Dictionary mapping calculated timestamps (int) to mDAQ data rows (list of str).
        - end_time_ms (int): The last calculated timestamp for mDAQ data (milliseconds).
        - processed_count (int): Number of data rows successfully processed and added to dict.

    Raises:
        NotADirectoryError: If mdaq_folder is not a valid directory.
        FileNotFoundError: If no suitable CSV files are found in the directory.
    """
    mdaq_data: Dict[int, List[str]] = {}
    processed_count = 0 # Initialize counter
    isi_accumulator_us = 0 # Accumulator for inter-sample intervals in microseconds
    logging.info(f'Starting mDAQ data processing in folder: {mdaq_folder}')
    start_process_time = time.time()

    if not os.path.isdir(mdaq_folder):
        raise NotADirectoryError(f"mDAQ folder not found or is not a directory: {mdaq_folder}")

    # Find CSV files named with digits (e.g., '1.csv', '10.csv')
    try:
        files = [f for f in os.listdir(mdaq_folder)
                 if f.endswith('.csv') and f[:-4].isdigit()]
        # Sort files numerically based on their names
        sorted_files = sorted(files, key=lambda x: int(x[:-4]))
    except Exception as e:
        logging.error(f"Error listing or sorting files in mdaq folder {mdaq_folder}: {e}")
        raise

    if not sorted_files:
        raise FileNotFoundError(f"No mDAQ CSV files (e.g., 1.csv, 2.csv) found in folder: {mdaq_folder}")

    logging.info(f"Found {len(sorted_files)} mDAQ files to process.")

    skipped_count = 0
    last_timestamp_ms = start_time_ms # Initialize with start time
    collision_check: Dict[int, int] = {} # To check for timestamp collisions

    # Process each file in numerical order
    for filename in tqdm(sorted_files, desc="Processing mDAQ files", unit="file"):
        file_path = os.path.join(mdaq_folder, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader) # Skip header row
                    logging.debug(f"mDAQ Header in {filename}: {header}")
                except StopIteration:
                    logging.warning(f"Skipping empty mDAQ file: {filename}")
                    continue # Skip empty files

                expected_columns = 16

                for row_idx, row in enumerate(reader):
                    if len(row) < 12:
                        logging.warning(f"Skipping row {row_idx + 2} in {filename}: Too few columns ({len(row)}). Expected at least 12. Line: {row}")
                        skipped_count += 1
                        continue

                    try:
                        # ISI is typically in column 12 (index 11)
                        isi_us = int(float(row[11]))
                        isi_accumulator_us += isi_us
                        # Calculate timestamp relative to BIOPAC start
                        timestamp_ms = start_time_ms + (isi_accumulator_us // 1000)

                        # Check for collisions (optional, for debugging)
                        if timestamp_ms in collision_check:
                            collision_check[timestamp_ms] += 1
                            if collision_check[timestamp_ms] == 2: # Log only first collision per timestamp
                                logging.warning(f"mDAQ timestamp collision detected for {timestamp_ms} ms in file {filename}, row approx {row_idx + 2}.")
                        else:
                            collision_check[timestamp_ms] = 1

                        last_timestamp_ms = timestamp_ms

                        # --- Data Extraction and Transformation ---
                        try:
                            ecg = row[0]
                            # EDA calculation: Resistance = 335544320 / raw_value
                            eda_raw_str = row[1]
                            eda_raw = float(eda_raw_str)
                            eda = str(eda_raw_str) #str(335544320 / eda_raw) if eda_raw != 0 else 'Infinity'
                        except (ValueError, IndexError) as e:
                            logging.warning(f"Skipping row {row_idx + 2} in {filename}: Error processing ECG/EDA raw value '{row[1]}'. Error: {e}. Row: {row}")
                            skipped_count += 1
                            continue
                        except ZeroDivisionError:
                             eda = 'Infinity'
                             logging.warning(f"EDA raw value is zero in row {row_idx + 2} in {filename}. Setting EDA to 'Infinity'. Row: {row}")

                        # Safely access other columns
                        ir = row[2] if len(row) > 2 else ''
                        red = row[3] if len(row) > 3 else ''
                        acc_x = row[4] if len(row) > 4 else ''
                        acc_y = row[5] if len(row) > 5 else ''
                        acc_z = row[6] if len(row) > 6 else ''
                        gyr_x = row[7] if len(row) > 7 else ''
                        gyr_y = row[8] if len(row) > 8 else ''
                        gyr_z = row[9] if len(row) > 9 else ''
                        batt = row[12] if len(row) > 12 else ''
                        rh = row[13] if len(row) > 13 else ''
                        amb_temp = row[14] if len(row) > 14 else ''
                        body_temp = row[15] if len(row) > 15 else ''

                        # Assemble the final data row
                        values = [
                            ecg, eda, ir, red, acc_x, acc_y, acc_z,
                            gyr_x, gyr_y, gyr_z, batt, rh, amb_temp, body_temp
                        ]

                        mdaq_data[timestamp_ms] = values
                        # processed_count incremented later based on dict size

                    except (ValueError, IndexError) as e:
                         logging.warning(f"Skipping row {row_idx + 2} in {filename}: Error processing ISI or calculating timestamp. Error: {e}. Row: {row}")
                         skipped_count += 1
                         continue # Skip row on error

        except FileNotFoundError:
            logging.error(f"mDAQ file disappeared during processing: {filename}")
        except Exception as e:
            logging.error(f"Error reading or processing mDAQ file {filename}: {e}")

    # Calculate final processed count based on the number of unique timestamps generated
    processed_count = len(mdaq_data)
    num_collisions = sum(count - 1 for count in collision_check.values() if count > 1)

    processing_time = time.time() - start_process_time
    logging.info(f'Finished mDAQ data processing. Time taken: {processing_time:.2f} seconds.')
    logging.info(f'mDAQ samples added to dictionary (unique timestamps): {processed_count}')
    logging.info(f'mDAQ rows skipped: {skipped_count}')
    if num_collisions > 0:
        logging.warning(f'mDAQ timestamp collisions detected (overwritten rows): {num_collisions}')
    else:
        logging.info('No mDAQ timestamp collisions detected.')


    if not mdaq_data:
        logging.warning("No mDAQ data was successfully processed.")
        return mdaq_data, start_time_ms, 0 # Return 0 processed count

    logging.info(f"mDAQ data range: {start_time_ms} ms to {last_timestamp_ms} ms")
    return mdaq_data, last_timestamp_ms, processed_count


def process_labels(label_file: str, time_offset: int = 0) -> Tuple[Dict[int, str], Dict[str, str]]:
    """
    Processes the label file (CSV or TXT) to extract timestamps and labels.
    Assumes timestamp is in the first column and label in the second.
    Optionally extracts metadata (device, session, subject, trial) from the first row if available.

    Args:
        label_file: Path to the label file.
        time_offset: An optional offset (in milliseconds) to add to each timestamp. Defaults to 0.

    Returns:
        A tuple containing:
        - labels (dict): Dictionary mapping adjusted timestamps (int) to labels (str).
        - metadata (dict): Dictionary containing metadata extracted from the first row.
    """
    labels: Dict[int, str] = {}
    metadata: Dict[str, str] = {}
    logging.info(f'Starting label data processing for: {label_file}')
    start_process_time = time.time()
    line_count = 0
    processed_count = 0

    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            # Detect delimiter (comma or tab)
            sniffer = csv.Sniffer()
            try:
                 # Read a larger sample for better sniffing, handle potential EOF
                sample = f.read(2048)
                if not sample:
                    logging.warning(f"Label file '{label_file}' appears to be empty.")
                    return labels, metadata # Return empty if file is empty
                dialect = sniffer.sniff(sample)
                f.seek(0)
                reader = csv.reader(f, dialect)
                logging.info(f"Detected delimiter: '{dialect.delimiter}'")
            except csv.Error:
                logging.warning("Could not detect delimiter, assuming comma.")
                f.seek(0)
                reader = csv.reader(f) # Default to comma

            for idx, row in enumerate(reader):
                line_count += 1
                if not row or len(row) < 2:
                    logging.warning(f"Skipping malformed or short row {idx + 1}: {row}")
                    continue

                try:
                    # Convert first column to timestamp in milliseconds
                    timestamp_ms = int(float(row[0])) + time_offset
                    label = row[1].strip() # Get label from second column

                    if not label:
                         logging.warning(f"Skipping row {idx + 1} due to empty label.")
                         continue

                    labels[timestamp_ms] = label
                    processed_count += 1

                    # Attempt to extract metadata from the first row (if columns exist)
                    if idx == 0 and len(row) >= 6:
                        metadata = {
                            # Use .get() with default empty string for safety
                            'device': row[2].strip() if len(row) > 2 else '',
                            'session': row[3].strip() if len(row) > 3 else '',
                            'subject': row[4].strip() if len(row) > 4 else '',
                            'trial': row[5].strip() if len(row) > 5 else ''
                        }
                        logging.info(f"Extracted metadata: {metadata}")

                except ValueError:
                    logging.warning(f"Skipping row {idx + 1} due to non-numeric timestamp: {row[0]}")
                except IndexError:
                    # This case is handled by the initial check `len(row) < 2` but kept for safety
                    logging.warning(f"Skipping row {idx + 1} due to missing columns: {row}")

    except FileNotFoundError:
        logging.error(f"Label file not found: {label_file}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing labels: {e}")
        raise

    processing_time = time.time() - start_process_time
    logging.info(f'Finished label data processing. Lines read: {line_count}. Labels processed: {processed_count}. Time taken: {processing_time:.2f} seconds.')
    if not labels:
        logging.warning("No labels were successfully processed.")
    return labels, metadata


# --- Data Writing Function ---

def write_data_efficient(biopac_data: Dict[int, List[str]], mdaq_data: Dict[int, List[str]],
                         labels: Dict[int, str], output_dir: str) -> Tuple[int, int]:
    """
    Writes the processed BIOPAC and mDAQ data, along with matched labels,
    to separate CSV files efficiently. Also generates a label assignment report.
    Returns the number of labels assigned to each dataset.

    Args:
        biopac_data: Dictionary mapping timestamps to BIOPAC data rows (ECG, PPG, SKT, EDA).
        mdaq_data: Dictionary mapping timestamps to mDAQ data rows.
        labels: Dictionary mapping timestamps to labels.
        output_dir: The directory to save the output CSV files.

    Returns:
        Tuple[int, int]: Number of labels assigned to BIOPAC, Number of labels assigned to mDAQ.
    """
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
    logging.info('Starting data and label assignment writing process')

    # Pre-sort timestamps for efficient searching and writing
    biopac_timestamps = sorted(biopac_data.keys())
    mdaq_timestamps = sorted(mdaq_data.keys())

    # --- Label Matching ---
    label_assignments = []
    biopac_label_matches: Dict[int, str] = {}
    mdaq_label_matches: Dict[int, str] = {}

    logging.info(f"Attempting to match {len(labels)} labels...")
    # Iterate through each label timestamp and find the nearest match in BIOPAC and mDAQ data
    for label_ts in sorted(labels.keys()):
        label = labels[label_ts]
        # Find nearest timestamp within a 1000ms window
        biopac_match = find_nearest_timestamp(label_ts, biopac_timestamps, window_ms=1000)
        mdaq_match = find_nearest_timestamp(label_ts, mdaq_timestamps, window_ms=1000)

        # If a match is found in either dataset, record the assignment
        if biopac_match or mdaq_match:
            biopac_row_index = biopac_timestamps.index(biopac_match) if biopac_match else None
            mdaq_row_index = mdaq_timestamps.index(mdaq_match) if mdaq_match else None

            assignment = {
                'original_ts': label_ts,
                'label': label,
                'biopac_ts': biopac_match,
                'mdaq_ts': mdaq_match,
                # Add 2 to index for 1-based row number + header row
                'biopac_row': biopac_row_index + 2 if biopac_row_index is not None else None,
                'mdaq_row': mdaq_row_index + 2 if mdaq_row_index is not None else None
            }
            label_assignments.append(assignment)

            # Store the label match for quick lookup during data writing
            if biopac_match:
                biopac_label_matches[biopac_match] = label
            if mdaq_match:
                mdaq_label_matches[mdaq_match] = label
    logging.info(f"Matched {len(label_assignments)} labels.")
    logging.info(f"Labels assigned to BIOPAC: {len(biopac_label_matches)}")
    logging.info(f"Labels assigned to mDAQ: {len(mdaq_label_matches)}")


    # --- CSV File Writing ---
    # Define headers for the output files
    biopac_header = ['timestamp_ms', 'ECG', 'PPG', 'SKT', 'EDA', 'label']
    mdaq_header = ['timestamp_ms', 'ecg', 'eda', 'ir', 'red', 'acc_x', 'acc_y', 'acc_z',
                   'gyr_x', 'gyr_y', 'gyr_z', 'batt%', 'relative_humidity', 'ambient_temp', 'body_temp', 'label']
    label_assignment_header = ['label', 'original_timestamp_ms', 'biopac_timestamp_ms',
                               'mdaq_timestamp_ms', 'biopac_row', 'mdaq_row', 'time_diff_ms']

    # Define output file paths
    biopac_output_path = os.path.join(output_dir, 'biopac_labels.csv')
    mdaq_output_path = os.path.join(output_dir, 'mdaq_labels.csv')
    label_output_path = os.path.join(output_dir, 'label_assignments.csv')

    logging.info(f"Writing BIOPAC data to: {biopac_output_path}")
    logging.info(f"Writing mDAQ data to: {mdaq_output_path}")
    logging.info(f"Writing Label Assignments to: {label_output_path}")

    try:
        with open(biopac_output_path, 'w', newline='', encoding='utf-8') as bf, \
             open(mdaq_output_path, 'w', newline='', encoding='utf-8') as mf, \
             open(label_output_path, 'w', newline='', encoding='utf-8') as lf:

            writers = {
                'biopac': csv.writer(bf),
                'mdaq': csv.writer(mf),
                'labels': csv.writer(lf)
            }

            # Write headers
            writers['biopac'].writerow(biopac_header)
            writers['mdaq'].writerow(mdaq_header)
            writers['labels'].writerow(label_assignment_header)

            # Write BIOPAC data rows with matched labels
            for ts in tqdm(biopac_timestamps, desc="Writing BIOPAC data", unit="row"):
                values = biopac_data.get(ts, ['','','','']) # Default to empty strings if ts somehow missing
                label = biopac_label_matches.get(ts, '') # Get label if timestamp matched, else empty string
                writers['biopac'].writerow([ts] + values + [label])

            # Write mDAQ data rows with matched labels
            mdaq_default = [''] * (len(mdaq_header) - 2) # timestamp and label excluded
            for ts in tqdm(mdaq_timestamps, desc="Writing mDAQ data", unit="row"):
                values = mdaq_data.get(ts, mdaq_default)
                label = mdaq_label_matches.get(ts, '') # Get label if timestamp matched, else empty string
                writers['mdaq'].writerow([ts] + values + [label])

            # Write label assignment details
            for assign in tqdm(label_assignments, desc="Writing Label Assignments", unit="label"):
                matched_ts = assign['biopac_ts'] or assign['mdaq_ts']
                time_diff = abs(assign['original_ts'] - matched_ts) if matched_ts is not None else None
                writers['labels'].writerow([
                    assign['label'],
                    assign['original_ts'],
                    assign['biopac_ts'] or '',
                    assign['mdaq_ts'] or '',
                    assign['biopac_row'] or '',
                    assign['mdaq_row'] or '',
                    time_diff if time_diff is not None else ''
                ])
    except IOError as e:
        logging.error(f"Error writing output files: {e}")
        raise # Re-raise the error after logging

    logging.info('Finished writing data and label assignments.')
    # Return label match counts for the final report
    return len(biopac_label_matches), len(mdaq_label_matches)


# --- Verification Function ---

def count_csv_rows(filepath: str) -> int:
    """Counts the number of data rows in a CSV file, excluding the header."""
    count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None) # Read/skip header
            if header:
                # Efficiently count remaining rows
                count = sum(1 for row in reader)
            else:
                logging.warning(f"File '{filepath}' appears empty or has no header.")
    except FileNotFoundError:
        logging.error(f"Verification Error: Output file not found: {filepath}")
        return -1 # Indicate error
    except Exception as e:
        logging.error(f"Verification Error: Could not read or count rows in {filepath}: {e}")
        return -1 # Indicate error
    return count

def verify_output_files(output_dir: str, original_biopac_count: int, original_mdaq_count: int) -> Dict:
    """
    Verifies the number of rows written to output CSV files against original counts.

    Args:
        output_dir: The directory containing the output CSV files.
        original_biopac_count: Number of BIOPAC data points processed (unique timestamps).
        original_mdaq_count: Number of mDAQ data points processed (unique timestamps).

    Returns:
        A dictionary containing verification results.
    """
    logging.info("--- Starting Output File Verification ---")
    verification_results = {
        'Original BIOPAC Points Processed (Unique TS)': original_biopac_count,
        'Original mDAQ Points Processed (Unique TS)': original_mdaq_count,
        'BIOPAC Output File Rows Written': 'N/A',
        'mDAQ Output File Rows Written': 'N/A',
        'BIOPAC Data Loss (Points)': 'N/A',
        'mDAQ Data Loss (Points)': 'N/A',
        'Verification Status': 'Errors Encountered' # Default status
    }

    biopac_output_path = os.path.join(output_dir, 'biopac_labels.csv')
    mdaq_output_path = os.path.join(output_dir, 'mdaq_labels.csv')

    biopac_output_count = count_csv_rows(biopac_output_path)
    mdaq_output_count = count_csv_rows(mdaq_output_path)

    biopac_loss = 'N/A'
    mdaq_loss = 'N/A'
    status = "Verification OK" # Assume OK initially

    if biopac_output_count != -1:
        verification_results['BIOPAC Output File Rows Written'] = biopac_output_count
        biopac_loss = original_biopac_count - biopac_output_count
        verification_results['BIOPAC Data Loss (Points)'] = biopac_loss
        if biopac_loss != 0:
            logging.warning(f"BIOPAC data loss detected: {biopac_loss} points (Processed Unique TS: {original_biopac_count}, Written: {biopac_output_count})")
            status = "Data Loss Detected"
        else:
             logging.info(f"BIOPAC row count verified: {biopac_output_count} rows written.")
    else:
        verification_results['BIOPAC Output File Rows Written'] = 'Error Reading File'
        status = "Errors Encountered" # Error reading file

    if mdaq_output_count != -1:
        verification_results['mDAQ Output File Rows Written'] = mdaq_output_count
        mdaq_loss = original_mdaq_count - mdaq_output_count
        verification_results['mDAQ Data Loss (Points)'] = mdaq_loss
        if mdaq_loss != 0:
            logging.warning(f"mDAQ data loss detected: {mdaq_loss} points (Processed Unique TS: {original_mdaq_count}, Written: {mdaq_output_count})")
            if status != "Errors Encountered": status = "Data Loss Detected" # Don't overwrite error status
        else:
            logging.info(f"mDAQ row count verified: {mdaq_output_count} rows written.")

    else:
         verification_results['mDAQ Output File Rows Written'] = 'Error Reading File'
         status = "Errors Encountered" # Error reading file

    verification_results['Verification Status'] = status
    logging.info(f"--- Verification Complete: {status} ---")
    return verification_results


# --- Main Execution Logic ---

def main():
    """
    Main function to orchestrate the data processing workflow including verification.
    Handles file/folder selection, processing, writing output, verification, and error handling.
    """
    global start_all # Allow modification of the global variable
    start_all = time.time() # Record the absolute start time

    output_dir = "" # Initialize output_dir
    base_output_dir = "" # Initialize base_output_dir
    output_folder = "" # Initialize output_folder
    log_handlers_closed = False # Flag to track if logger was shut down

    # --- Setup ---
    try:
        # Select input/output paths using GUI dialogs
        biopac_file, mdaq_folder, label_file, output_folder = select_folders()

        # Create a unique base output directory for this run
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = os.path.join(output_folder, f"merged_data_{timestamp_str}")
        os.makedirs(base_output_dir, exist_ok=True)

        # Initialize logging within the base output directory
        setup_logger(base_output_dir)
        logging.info(f"Logger initialized. Log file: {os.path.join(base_output_dir, 'processing.log')}")
        output_dir = base_output_dir # Start with base dir as the output dir

        logging.info("--- Starting Data Processing ---")
        logging.info(f"BIOPAC File: {biopac_file}")
        logging.info(f"mDAQ Folder: {mdaq_folder}")
        logging.info(f"Label File: {label_file}")
        logging.info(f"Base Output Directory: {output_dir}")

        # --- Data Processing ---
        # Process BIOPAC first to get the absolute start time and data
        # Capture the processed count (unique timestamps)
        biopac_start_ms, biopac_end_ms, biopac_data, biopac_processed_count = process_biopac(biopac_file)

        # Process mDAQ using the BIOPAC start time
        # Capture the processed count (unique timestamps)
        mdaq_data, mdaq_end_ms, mdaq_processed_count = process_mdaq(mdaq_folder, biopac_start_ms)

        # Process labels (no time offset needed as BIOPAC provides absolute start)
        labels, metadata = process_labels(label_file, time_offset=0)

        # --- Finalize Output Directory Name ---
        # Attempt to rename output directory using subject/session from metadata
        final_log_target_dir = base_output_dir # Default target for logger re-init

        if metadata.get('subject') and metadata.get('session'):
            try:
                # Sanitize subject/session names for use in directory path
                subject_sanitized = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in metadata['subject'])
                session_sanitized = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in metadata['session'])

                # Construct the desired metadata-based directory name
                metadata_output_dir = os.path.join(output_folder,
                                                  f"merged_{subject_sanitized}_{session_sanitized}_{timestamp_str}")

                if not os.path.exists(metadata_output_dir):
                    # --- Close existing log handlers BEFORE renaming ---
                    logging.info(f"Attempting to rename output directory to: {metadata_output_dir}")
                    log_handlers = logging.getLogger().handlers[:] # Get a copy
                    for handler in log_handlers:
                        if isinstance(handler, logging.FileHandler):
                            handler.close() # Close the file stream
                            logging.getLogger().removeHandler(handler) # Remove handler from logger
                    log_handlers_closed = True # Mark that we closed handlers
                    time.sleep(0.1) # Allow OS time to release handle

                    # --- Attempt rename ---
                    os.rename(base_output_dir, metadata_output_dir)
                    output_dir = metadata_output_dir # Update output_dir to the new name
                    final_log_target_dir = metadata_output_dir # Log should go here now
                    print(f"Output directory successfully renamed to: {output_dir}") # Use print as logger is closed

                else:
                     logging.warning(f"Metadata-based directory '{metadata_output_dir}' already exists. Using default: {base_output_dir}")

            except OSError as e:
                 err_msg = f"Could not rename output directory with metadata: {e}. Using default: {base_output_dir}"
                 if log_handlers_closed: print(f"ERROR: {err_msg}")
                 else: logging.error(err_msg)
            except Exception as e: # Catch other potential errors
                 err_msg = f"Unexpected error during directory renaming check: {e}. Using default: {base_output_dir}"
                 if log_handlers_closed: print(f"ERROR: {err_msg}")
                 else: logging.error(err_msg)
            finally:
                # --- Re-initialize logger AFTER rename attempt ---
                if log_handlers_closed:
                    setup_logger(final_log_target_dir)
                    logging.info(f"Logging re-initialized in final directory: {final_log_target_dir}")
                    log_handlers_closed = False # Reset flag
        else:
            logging.info("No subject/session metadata found in labels file to rename output directory.")

        # --- Write Output Files ---
        # Pass the potentially renamed output_dir
        # Capture label match counts
        biopac_labels_assigned, mdaq_labels_assigned = write_data_efficient(biopac_data, mdaq_data, labels, output_dir)

        # --- Verification Step ---
        verification_results = verify_output_files(output_dir, biopac_processed_count, mdaq_processed_count)

        # --- Write Final Report ---
        logging.info("Generating final performance and verification report...")
        total_time_sec = time.time() - start_all
        try:
            process = psutil.Process(os.getpid())
            peak_memory_mb = process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            logging.warning(f"Could not get memory usage: {e}")
            peak_memory_mb = "N/A"

        # Combine processing stats and verification results
        final_report_data = {
            'Processing Summary': {
                'Total Processing Time': f"{total_time_sec:.2f}s",
                'Peak Memory Usage (RSS)': f"{peak_memory_mb:.2f}MB" if isinstance(peak_memory_mb, float) else peak_memory_mb,
                'Labels Input': len(labels),
                'Labels Assigned to BIOPAC': biopac_labels_assigned,
                'Labels Assigned to mDAQ': mdaq_labels_assigned,
            },
            'File Verification': verification_results # Add the dict returned by verify_output_files
        }

        metrics_path = os.path.join(output_dir, 'performance_metrics.txt')
        try:
            with open(metrics_path, 'w', encoding='utf-8') as f:
                f.write("Performance and Verification Report\n")
                f.write("="*50 + "\n")
                for section, metrics in final_report_data.items():
                    f.write(f"\n{section}:\n")
                    f.write("-" * len(section) + "\n")
                    for key, value in metrics.items():
                        f.write(f"{key}: {value}\n")
            logging.info(f"Final report written to: {metrics_path}")
        except IOError as e:
             logging.error(f"Could not write final report to {metrics_path}: {e}")


        # --- Final Success Message ---
        status_message = f"Data processed successfully!\nVerification Status: {verification_results.get('Verification Status', 'Unknown')}"
        status_message += f"\nOutput directory: {output_dir}"
        # Use warning icon if data loss detected
        msg_icon = messagebox.WARNING if verification_results.get('Verification Status') == "Data Loss Detected" else messagebox.INFO
        messagebox.showinfo("Processing Complete", status_message, icon=msg_icon)
        logging.info('--- Processing completed ---')


    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        # Handle specific known errors (file/folder selection, header parsing)
        logging.error(f"Configuration or Input File Error: {e}", exc_info=True) # Add traceback
        messagebox.showerror("Input Error", f"Processing failed:\n{str(e)}\n\nPlease check file selections and BIOPAC/Label file formats.")
    except Exception as e:
        # Handle unexpected errors during processing
        logging.exception('An unexpected error occurred during processing.') # Log full traceback
        # Attempt to show error message, include output_dir if it was created
        err_msg = f"An unexpected error occurred:\n{type(e).__name__}: {str(e)}"
        log_file_path = os.path.join(output_dir, 'processing.log') if output_dir else "processing.log"
        if output_dir and os.path.exists(output_dir):
             err_msg += f"\n\nPartial results and logs might be in:\n{output_dir}"
             err_msg += f"\nCheck '{log_file_path}' for details."
        else:
             err_msg += f"\n\nCheck console or log file ('{log_file_path}' if created) for details."
        messagebox.showerror("Processing Error", err_msg)
    finally:
        # Ensure Tkinter root is destroyed if it exists to prevent hanging processes
        try:
            # Check if a default root exists and destroy it
            root = tk._default_root
            if root and root.winfo_exists():
                root.destroy()
        except Exception as e:
            # Use print as logger might be unreliable here
            print(f"Debug: Error destroying Tkinter root window during cleanup: {e}")
        logging.info('--- Script finished ---')


if __name__ == "__main__":
    main()
    logging.info("--- Script completed ---")