import os
import errno
import pandas as pd
import numpy as np
from typing import Optional, Union


class Table:
    def __init__(
        self,
        data: np.ndarray,
        table_name: Optional[str] = "",
        header: Optional[list] = None,
        index: Optional[list] = None,
    ) -> None:
        """
        ### Description:
        - A class used to manage the printing and saving of 2d numpy arrays.
        ---
        ### Args:
        - data: the 2d numpy array to use in the table.
        ---
        ### Optional Args:
        - table_name (str): the name of the table.
        - header (list): list of the columns names.
        - index (list): list of the row names.
        """
        # expand dims if the data is 1-dimensional
        if data.ndim == 1:
            self.data = np.expand_dims(data, axis=0)
        elif data.ndim == 2:
            self.data = data
        else:
            raise Exception(
                f"The data has the wrong number of dimensions,"
                f" use 1d or 2d array (ndim = {data.ndim})"
            )
        row_num, col_num = self.data.shape

        self.table_name = table_name

        # create index if missing
        if index == None:
            index = [f"Row [{i}]" for i in range(row_num)]
        # check if the index is valid
        if len(index) != row_num:
            raise Exception(
                f"The index must have the same number of elements as the rows in data\n"
                f"index = {len(index)}\n"
                f"rows = {row_num}\n"
            )
        self.index = index

        # create header if missing
        if header == None:
            header = [f"Col [{i}]" for i in range(col_num)]
        # check if the header is valid
        if len(header) != col_num:
            raise Exception(
                f"The header must have the same number of elements as the columns in data\n"
                f"header = {len(header)}\n"
                f"columns = {col_num}\n"
            )
        self.header = header

        # other options
        pd.options.display.float_format = "{: 6.3f} |".format
        pd.options.display.max_columns = col_num
        self.max_width = 100

    def print(
        self,
        rows: Optional[list] = None,
        columns: Optional[list] = None,
        header: Optional[list] = None,
        to_string: Optional[bool] = False,
    ) -> Union[None, str]:
        """
        ### Description:
        - Print the selected rows and columns of the table.
        ---
        ### Args:
        - rows (list): list of the rows to print.
                        If rows is None then all the rows will be printed.
        - columns (list): list of the columns to print.
                        If columns is None then all the columns will be printed.
        - header (list): list of the columns names.
                        If header is None then the default value will be used.
        - to_string (bool): if set to True returns the table as a string instead of printing it.
        """
        # check if the inputs are valid and format the data
        rows = rows if rows != None else list(range(self.data.shape[0]))
        columns = columns if columns != None else list(range(self.data.shape[1]))
        if header != None:
            if len(header) != len(columns):
                raise Exception(
                    "header and columns must have the same number of elements\n"
                    f"header = {len(header)}\n"
                    f"columns = {len(columns)}\n"
                )
            header = [f"{i} |" for i in header]
        else:
            header = [f"{self.header[i]} |" for i in columns]
        index = [self.index[i] for i in rows]

        # create dataframe
        dataframe = pd.DataFrame(
            self.data[rows][:, columns],
            columns=header,
            index=index,
        )

        width = len(dataframe.to_string().splitlines()[0])
        if width > self.max_width:
            width = self.max_width

        # print the table or return it
        with pd.option_context("display.width", width + 2):
            string_table = f"{f' {self.table_name} ':-^{width}}" + "\n"
            string_table += dataframe.to_string() + "\n"
            string_table += f"{f'':-^{width}}"
            if to_string:
                return string_table
            else:
                print(string_table)

    def print_ms(
        self,
        ms_to_print: list,
        framerate: float,
        to_string: Optional[bool] = False,
    ) -> Union[None, str]:
        """
        ### Description:
        - Print the frames (columns) at specified times.
        ---
        ### Args:
        - ms_to_print (list): list of the times to print.
        - framerate (float): framerate of the data.
        - to_string (bool): if set to True returns the table as a string instead of printing it.
        ---
        ### Example:
        - ms_to_print = [80, 160, 320, 400, 560, 1000]
        - framerate = 25 fps

        the printed frames (columns) will be: [1, 3, 5, 7, 11, 24].
        """
        frame_ms = 1000 / framerate
        frames_to_print = [int(x // frame_ms - 1) for x in ms_to_print]
        header = [f"{ms} ms" for ms in ms_to_print]

        # if the aren't that many frames in the data skip the extra ones
        temp_frames_to_print = []
        for frame in frames_to_print:
            if frame >= len(self.header):
                break
            else:
                temp_frames_to_print.append(frame)
        frames_to_print = temp_frames_to_print
        header = header[slice(len(frames_to_print))]

        return self.print(columns=frames_to_print, header=header, to_string=to_string)

    def to_csv(self, path: str, file_name: Optional[str] = None) -> None:
        """
        ### Description:
        - Save the table to a .csv file.
        ---
        ### Args:
        - path (str): the path of the directory to put the .csv file.
        - file_name (str): the name of the .csv. If not passed, table_name will be used as file_name.
        """

        # create a valid name
        if file_name != None:
            if not file_name.endswith(".csv"):
                file_name = file_name + ".csv"
        else:
            if self.table_name == "":
                file_name = "table.csv"
            else:
                file_name = f"{self.table_name.replace(' ', '_')}.csv"

        # create directories in the path if not already present
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        # append the csv name to the path
        csv_path = os.path.join(path, file_name)

        # save csv
        dataframe = pd.DataFrame(self.data, columns=self.header, index=self.index)
        dataframe.to_csv(csv_path)
