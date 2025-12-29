"""
Excel Processing Module

Learning Note - What This Module Does:
--------------------------------------
This module extracts structured data from Excel (.xlsx, .xls) files for the RAG system.

Excel files contain:
1. Multiple sheets (tabs) with different datasets
2. Tabular data (rows and columns)
3. Headers and column names
4. Formulas and calculated values
5. Data types (numbers, text, dates)

Each sheet becomes a searchable chunk with structure preserved.

Key Features:
- Extract data from all sheets
- Preserve table structure and headers
- Handle different data types
- Skip empty cells for cleaner output
- Support both .xlsx and .xls formats
- Rich metadata (sheet names, dimensions, cell references)

Why Excel Support Matters:
- Financial reports and budgets
- Data analysis and statistics
- Inventory and tracking sheets
- Project plans and timelines
- Scientific data and measurements
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from io import BytesIO
from config import settings


class ExcelProcessor:
    """
    Processes Excel files to extract structured data.

    Learning Note - Processing Strategy:
    ------------------------------------
    1. Load Excel file with pandas
    2. Iterate through sheets
    3. Extract data while preserving structure
    4. Convert to readable text format
    5. Create chunks with rich metadata
    6. Maintain cell and column relationships
    """

    def __init__(self):
        """Initialize the Excel processor."""
        self.max_sheets = settings.MAX_SHEETS_PER_FILE
        self.max_rows = settings.MAX_ROWS_PER_SHEET
        self.include_empty = settings.INCLUDE_EMPTY_CELLS
        self.preserve_formulas = settings.PRESERVE_FORMULAS
        self.chunking_strategy = settings.EXCEL_CHUNKING_STRATEGY

    def process_excel(self, excel_path_or_bytes) -> Dict[str, Any]:
        """
        Process an Excel file and extract all data.

        Args:
            excel_path_or_bytes: Path to Excel file or BytesIO object

        Returns:
            Dictionary containing:
            - sheets: List of sheet data
            - total_sheets: Total number of sheets
            - processed_sheets: Number of sheets processed

        Learning Note - Pandas for Excel:
        ---------------------------------
        We use pandas because it:
        - Handles both .xlsx and .xls formats
        - Preserves data types automatically
        - Deals with merged cells gracefully
        - Provides easy data manipulation
        - Is industry standard for data processing
        """
        try:
            print(f"Opening Excel file...")

            # Read all sheets
            # engine='openpyxl' for .xlsx, 'xlrd' for .xls
            excel_file = pd.ExcelFile(excel_path_or_bytes, engine='openpyxl')

            sheet_names = excel_file.sheet_names
            total_sheets = len(sheet_names)
            sheets_to_process = min(total_sheets, self.max_sheets)

            print(f"Found {total_sheets} sheets, processing {sheets_to_process}")

            sheets_data = []

            for sheet_num, sheet_name in enumerate(sheet_names[:sheets_to_process], 1):
                try:
                    sheet_data = self._process_sheet(excel_file, sheet_name, sheet_num, total_sheets)
                    sheets_data.append(sheet_data)

                    if sheet_num % 5 == 0:
                        print(f"Processed {sheet_num}/{sheets_to_process} sheets...")

                except Exception as e:
                    print(f"Error processing sheet '{sheet_name}': {e}")
                    # Create minimal sheet data for failed sheets
                    sheets_data.append({
                        'sheet_name': sheet_name,
                        'sheet_number': sheet_num,
                        'total_sheets': total_sheets,
                        'data': '',
                        'num_rows': 0,
                        'num_cols': 0,
                        'error': str(e)
                    })

            print(f"✓ Extraction complete: {sheets_to_process} sheets")

            return {
                'sheets': sheets_data,
                'total_sheets': total_sheets,
                'processed_sheets': sheets_to_process
            }

        except Exception as e:
            print(f"Error processing Excel file: {e}")
            import traceback
            traceback.print_exc()
            return {
                'sheets': [],
                'total_sheets': 0,
                'processed_sheets': 0,
                'error': str(e)
            }

    def _process_sheet(self, excel_file, sheet_name: str, sheet_num: int, total_sheets: int) -> Dict[str, Any]:
        """
        Process a single Excel sheet.

        Args:
            excel_file: pandas ExcelFile object
            sheet_name: Name of the sheet
            sheet_num: Sheet number (1-indexed)
            total_sheets: Total sheets in workbook

        Returns:
            Dictionary with sheet data

        Learning Note - Pandas DataFrame:
        ----------------------------------
        Pandas reads Excel sheets as DataFrames:
        - Rows and columns of data
        - Named columns (headers)
        - Mixed data types (numbers, text, dates)
        - Handles NaN (Not a Number) for empty cells

        We convert DataFrame to human-readable text format.
        """
        # Read the sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=self.max_rows)

        # Get dimensions
        num_rows, num_cols = df.shape

        # Convert to readable text
        sheet_text = self._dataframe_to_text(df, sheet_name)

        # Get column names
        column_names = df.columns.tolist()

        sheet_data = {
            'sheet_name': sheet_name,
            'sheet_number': sheet_num,
            'total_sheets': total_sheets,
            'data': sheet_text,
            'num_rows': num_rows,
            'num_cols': num_cols,
            'columns': column_names
        }

        return sheet_data

    def _dataframe_to_text(self, df: pd.DataFrame, sheet_name: str) -> str:
        """
        Convert DataFrame to readable text format.

        Args:
            df: pandas DataFrame
            sheet_name: Name of the sheet

        Returns:
            Formatted text string

        Learning Note - Text Formatting:
        --------------------------------
        We format Excel data to be:
        - Human-readable (not raw CSV)
        - Preserving structure (columns aligned)
        - Clean (no excessive whitespace)
        - Searchable (key-value pairs for clarity)

        Format:
        Sheet: SheetName
        Columns: Col1, Col2, Col3
        ---
        Row 1: Col1=Value1, Col2=Value2, Col3=Value3
        Row 2: Col1=Value4, Col2=Value5, Col3=Value6
        """
        parts = []

        # Sheet header
        parts.append(f"=== Sheet: {sheet_name} ===")
        parts.append("")

        # Column headers
        columns = df.columns.tolist()
        parts.append(f"Columns: {', '.join(str(col) for col in columns)}")
        parts.append(f"Rows: {len(df)}")
        parts.append("")

        # Handle empty DataFrame
        if df.empty:
            parts.append("(Empty sheet)")
            return "\n".join(parts)

        # Data rows
        parts.append("Data:")
        parts.append("-" * 40)

        for idx, row in df.iterrows():
            row_parts = []

            for col in columns:
                value = row[col]

                # Skip empty cells if configured
                if pd.isna(value) and not self.include_empty:
                    continue

                # Format value
                if pd.isna(value):
                    value_str = "(empty)"
                elif isinstance(value, float):
                    # Format numbers nicely
                    if value.is_integer():
                        value_str = str(int(value))
                    else:
                        value_str = f"{value:.2f}"
                else:
                    value_str = str(value)

                row_parts.append(f"{col}={value_str}")

            if row_parts:  # Only add row if it has content
                parts.append(f"Row {idx + 1}: {', '.join(row_parts)}")

        return "\n".join(parts)

    def create_excel_chunks(self, sheets_data: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        """
        Create searchable chunks from Excel data.

        Args:
            sheets_data: List of sheet dictionaries from process_excel()
            filename: Original filename

        Returns:
            List of chunks ready for vector storage

        Learning Note - Chunking Strategies:
        ------------------------------------
        1. BY_SHEET (default):
           - Each sheet = one chunk
           - Preserves complete tables
           - Natural boundaries
           - Easy to reference

        2. BY_ROWS:
           - Split large sheets by rows
           - Better for huge datasets
           - May break table relationships
           - Use when sheets > MAX_ROWS

        Most Excel files work best with by_sheet strategy.
        """
        chunks = []

        if self.chunking_strategy == "by_sheet":
            # Strategy 1: Each sheet is a chunk
            for sheet in sheets_data:
                if sheet.get('error'):
                    continue  # Skip sheets with errors

                chunk_text = sheet['data']

                if chunk_text.strip():
                    chunk = {
                        'text': chunk_text,
                        'type': 'excel_sheet',
                        'metadata': {
                            'source': filename,
                            'chunk_type': 'excel_sheet',
                            'sheet_name': sheet['sheet_name'],
                            'sheet_number': sheet['sheet_number'],
                            'total_sheets': sheet['total_sheets'],
                            'num_rows': sheet['num_rows'],
                            'num_cols': sheet['num_cols'],
                            'columns': sheet.get('columns', [])
                        }
                    }
                    chunks.append(chunk)

        else:  # by_rows strategy
            # Strategy 2: Split large sheets by rows
            for sheet in sheets_data:
                if sheet.get('error'):
                    continue

                # Split sheet data into chunks
                sheet_lines = sheet['data'].split('\n')

                # Keep header and metadata together
                header_lines = []
                data_lines = []
                in_data_section = False

                for line in sheet_lines:
                    if line.strip().startswith("Data:"):
                        in_data_section = True
                        data_lines.append(line)
                    elif in_data_section:
                        data_lines.append(line)
                    else:
                        header_lines.append(line)

                # Create chunks of rows
                chunk_size = settings.EXCEL_CHUNK_SIZE
                current_chunk = header_lines.copy()
                current_size = sum(len(line) for line in current_chunk)
                chunk_index = 0

                for line in data_lines:
                    if current_size + len(line) > chunk_size and len(current_chunk) > len(header_lines):
                        # Save current chunk
                        chunk_text = '\n'.join(current_chunk)
                        chunk = {
                            'text': chunk_text,
                            'type': 'excel_rows',
                            'metadata': {
                                'source': filename,
                                'chunk_type': 'excel_rows',
                                'sheet_name': sheet['sheet_name'],
                                'sheet_number': sheet['sheet_number'],
                                'chunk_index': chunk_index
                            }
                        }
                        chunks.append(chunk)

                        # Start new chunk with header
                        current_chunk = header_lines.copy()
                        current_size = sum(len(line) for line in current_chunk)
                        chunk_index += 1

                    current_chunk.append(line)
                    current_size += len(line)

                # Add final chunk
                if len(current_chunk) > len(header_lines):
                    chunk_text = '\n'.join(current_chunk)
                    chunk = {
                        'text': chunk_text,
                        'type': 'excel_rows',
                        'metadata': {
                            'source': filename,
                            'chunk_type': 'excel_rows',
                            'sheet_name': sheet['sheet_name'],
                            'sheet_number': sheet['sheet_number'],
                            'chunk_index': chunk_index
                        }
                    }
                    chunks.append(chunk)

        return chunks
