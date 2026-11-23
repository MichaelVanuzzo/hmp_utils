import os
import pandas as pd
import numpy as np
from typing import Optional, Union

from .. import mkdir_if_missing

MAX_WIDTH = 120


class TableManager:
    """
    Class used to manage the printing and saving of 2d numpy arrays.

    """

    def __init__(
        self,
        data: np.ndarray,
        table_name: str = "",
        header: Optional[list] = None,
        index: Optional[list] = None,
    ) -> None:
        """
        Initialize the TableManager class.

        ---
        ### Args:
        - `data` (np.ndarray): the 2d numpy array to use in the table.
        - `table_name` (str): the name of the table.
        - `header` (list)(optional): list of the columns names.
        - `index` (list)(optional): list of the row names.
        """
        # expand dims if the data is 1-dimensional
        if data.ndim == 1:
            self.data = np.expand_dims(data, axis=0)
        elif data.ndim == 2:
            self.data = data
        else:
            raise Exception(f"The data has the wrong number of dimensions, use 1d or 2d array (ndim = {data.ndim})")

        self.table_name = table_name

        row_num, col_num = self.data.shape

        # create index if missing
        if index is None:
            self.index = [f"Row [{i}]" for i in range(row_num)]
        else:
            self.index = index
        if len(self.index) != row_num:
            raise ValueError(f"Index length ({len(self.index)}) must match number of rows ({row_num})")

        # create header if missing
        if header is None:
            self.header = [f"Col [{i}]" for i in range(col_num)]
        else:
            self.header = header
        if len(self.header) != col_num:
            raise ValueError(f"Header length ({len(self.header)}) must match number of columns ({col_num})")

        # pd options
        pd.options.display.float_format = "{: 6.3f} |".format
        pd.options.display.max_columns = col_num
        self.max_width = MAX_WIDTH

    def print(
        self,
        rows: Optional[list] = None,
        columns: Optional[list] = None,
        to_string: Optional[bool] = False,
    ) -> Union[None, str]:
        """
        Print the table.

        ---
        ### Args:
        - `rows` (list)(optional): list of the rows to print. If rows is None then all the rows will be printed.
        - `columns` (list)(optional): list of the columns to print. If columns is None then all the columns will be printed.
        - `to_string` (bool)(optional): if True return the table as a string.
        ---
        ### Returns:
        - `None` if to_string is False.
        - `str` if to_string is True.
        """
        # check if the inputs are valid and format the data
        rows = rows if rows is not None else list(range(self.data.shape[0]))
        columns = columns if columns is not None else list(range(self.data.shape[1]))
        index = [self.index[i] for i in rows]
        header = [self.header[i] for i in columns]

        # create dataframe
        dataframe = pd.DataFrame(
            self.data[np.ix_(rows, columns)],
            columns=header,
            index=index,
        )

        width = len(dataframe.to_string().splitlines()[0])
        width = min(width, self.max_width)
        # print the table or return it
        with pd.option_context("display.width", width + 2):
            string_table = f"{f' {self.table_name} ':-^{width}}" + "\n"
            string_table += dataframe.to_string() + "\n"
            string_table += f"{f'':-^{width}}"
            if to_string:
                return string_table
            else:
                print(string_table)

    def to_csv(self, path: str, file_name: Optional[str] = None) -> None:
        """
        Save the table as a csv file.

        ---
        ### Args:
        - `path` (str): the path where to save the csv.
        - `file_name` (str)(optional): the name of the csv file. If file_name is None then the table_name will be used.
        """

        # create a valid name
        if file_name is not None:
            if not file_name.endswith(".csv"):
                file_name = file_name + ".csv"
        elif self.table_name == "":
            file_name = "table.csv"
        else:
            file_name = f"{self.table_name.replace(' ', '_')}.csv"

        # create directories in the path if not already present
        mkdir_if_missing(path)

        # append the csv name to the path
        csv_path = os.path.join(path, file_name)

        # save csv
        dataframe = pd.DataFrame(self.data, columns=self.header, index=self.index)
        dataframe.to_csv(csv_path)
        print(f"Saved {self.table_name} to {csv_path}")
