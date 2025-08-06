#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/26 15:21
# @Author  : oliver
# @File    : reporter.py
# @Software: PyCharm
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :TrackerCore.py
# @Time      :2023/12/14 15:29
# @Author    :Oliver
import os
import platform
import re
import random
from datetime import datetime
import sys, socket, select
from io import StringIO, TextIOWrapper
import importlib.util
from random import shuffle

# from config import get_str_time
from .util import StrNumberIdIterator, StateEnum, get_str_time
"""
管理输出的接口,包括日志，md文件（图文）和输出控制台
同时也管理交互，负责接受输入的指令
"""

class Reporter(object):
    _main_id_pool = StrNumberIdIterator(0, 10, shuffle=False)
    _handles_id_pool = StrNumberIdIterator(0, 100, shuffle=False)
    def __init__(self, args, root_dir: str, save_dir:str, prefix=None,
                 flag_split_log=False, flag_debug_mode=False, *, build_time=None, _platform=None):
        """
        :param args:  将传递给terminalHandle，涉及到部分状态的信息
        :param root_dir: 所有报告保存的根目录，不能修改
        :param save_dir: 所有报告保存的根目录名称，这个名称可以修改
        :param flag_split_log:主要是log是否会根据不同的子reporter（每次band会产生一个），额外产生一个分离的文件。 除了这里，绑定时也可以指定该量
        :param flag_debug_mode: debug mode 下 不会真的保存到log和md中，只有stdout在工作
        """
        self.args = args
        self.root_dir = root_dir
        self.report_base_name = save_dir
        self.build_time = datetime.now().strftime("%Y%m%d_%H-%M-%S") if build_time is None else build_time
        self.platform = platform.system() if _platform is None else _platform
        self.prefix = prefix
        self.flag_split_log = flag_split_log
        self.flag_debug_mode = flag_debug_mode

        self.state = StateEnum.INITIALIZING
        self.handles = {}
        self.terminal_ctrl = TerminalControl(args)
        # self.tb_summary_writer = self.terminal_ctrl.get_tb_summary_writer()
        self._md_hd:list[MarkdownHandle] = []
        self._log_hd:list[PlainTextHandle]  = []
        self._csv_hd:list[CSVHandle] = []

        if self._is_main():
            self._main_id = Reporter._main_id_pool.pop()
            self.prefix = prefix + f"#{self._main_id}" if prefix is not None else f'main#{self._main_id}'
            self.sub_reporter = {self.prefix:self}
            self.sub_reporter_id_pool = StrNumberIdIterator(0, 2000, shuffle=True)

        self.init_handle()
        self.state = StateEnum.ACTIVE
        if self.flag_debug_mode:
            self("[!]Note: The Reporter is in debug mode and will not save logs!")

    @property
    def report_path(self):
        return os.path.join(self.root_dir, "{}_{}_{}".format(self.build_time, self.report_base_name, self.platform))

    def init_handle(self):
        self.add_txt_handle("log_main.log", self.prefix, True)
        self.add_md_handle("log_main.md", self.prefix)
        self.add_csv_handle("loss_main.csv", self.prefix)
        if self.flag_split_log:
            self.add_txt_handle(f"log_{self.prefix}.log", self.prefix, True)
            self.add_md_handle(f"log_{self.prefix}.md", self.prefix)
            self.add_csv_handle(f"loss_{self.prefix}.csv", self.prefix)


    def add_md_handle(self, filename, prefix)->None:
        if self.flag_debug_mode:
            return
        hd_id = self._handles_id_pool.pop()
        prefix = prefix+f".md_hd#{hd_id}"
        handle = MarkdownHandle(self.args, self.report_path, prefix, filename)
        self.handles.update({prefix:handle})
        self._md_hd.append(handle)
        return

    def add_txt_handle(self, filename, prefix, log_mode=False)->None:
        if self.flag_debug_mode:
            return
        hd_id = self._handles_id_pool.pop()
        prefix = prefix + f".txt_hd#{hd_id}"
        handle = PlainTextHandle(self.args, self.report_path, prefix, filename, log_mode)
        self.handles.update({prefix: handle})
        self._log_hd.append(handle)
        return

    def add_csv_handle(self, filename, prefix):
        if self.flag_debug_mode:
            return
        hd_id = self._handles_id_pool.pop()
        prefix = prefix + f".csv_hd#{hd_id}"
        handle = CSVHandle(self.args, self.report_path, prefix, filename)
        self.handles.update({prefix: handle})
        self._csv_hd.append(handle)
        return

    def band(self, display_name:str, flag_split_log=None):
        # for prefix, sub_report in self.sub_reporter.items():
        #     if sub_report.state == StateEnum.FINISHED:
        #         self.sub_reporter.pop(prefix)
        #         self.sub_reporter_id_pool.release(prefix)
        self.check_sub_reporter()
        flag_split_log = flag_split_log if flag_split_log is not None else self.flag_split_log
        display_name = display_name + f"#{self.sub_reporter_id_pool.pop()}"
        reporter = SubReporter(self.args, self.root_dir, self.report_base_name, prefix=display_name,
                               flag_split_log=flag_split_log, flag_debug_mode=self.flag_debug_mode,
                               build_time=self.build_time, _platform=self.platform)
        self.sub_reporter.update({display_name:reporter})
        return reporter

    @property
    def _used_sub_reporter_name(self):
        return self.sub_reporter.keys()
    @property
    def _used_handle_name(self):
        return self.handles.keys()
    @property
    def prefix_id(self):
        return re.search("#[0-9]*$",self.prefix).group()

    def check_sub_reporter(self):
        to_remove = [
            prefix
            for prefix, sub_report in self.sub_reporter.items()
            if sub_report.state == StateEnum.FINISHED
        ]
        for prefix in to_remove:
            self.sub_reporter.pop(prefix)
            self.sub_reporter_id_pool.release(prefix)

    def close(self):
        if self._is_main():
            for prefix, sub_reporter in self.sub_reporter.items():
                if prefix !=self.prefix:
                    sub_reporter.close()
                    self.sub_reporter_id_pool.release(sub_reporter.prefix_id)
            self._main_id_pool.release(self._main_id)
        for name, handle in self.handles.items():
            handle.close()
            self._handles_id_pool.release(handle.handle_id)
        self.state = StateEnum.FINISHED

    def _is_main(self):
        return True

    def check_input(self):
        return self.terminal_ctrl.check_input(True)

    def add_data_markdown_word(self, data: str):
        for md_hd in self._md_hd:
            md_hd.add_line(data, ending="")

    def add_data_log(self, data: str):
        for log_hd in self._log_hd:
            log_hd.add_line(data, ending="")

    def add_line_csv(self, data_line: list):
        for csv_hd in self._csv_hd:
            csv_hd.add_raw(data_line)

    def add_line_stdout(self, data_line):
        self.terminal_ctrl.to_stdout(data_line)

    def save_figure_to_file(self, fig, save_name):
        if self.flag_debug_mode:
            return "debugging"
        md_hd = self._md_hd[0]
        rel_save_path = md_hd.save_figure_only(fig, save_name)
        for log_hd in self._log_hd:
            log_hd.add_line(f"saved a figure in {rel_save_path}")
        return rel_save_path

    def add_figure_by_data(self, fig, save_name=None,  title="", describe="", alt_text=None,):
        if self.flag_debug_mode:
            return "debugging"
        md_hd = self._md_hd[0]
        rel_save_path = md_hd.save_figure_only(fig, save_name)
        for md_hd in self._md_hd:
            md_hd.add_figure_by_path(rel_save_path, alt_text, title, describe)
        for log_hd in self._log_hd:
            log_hd.add_line(f"saved a figure in {rel_save_path}")
        return rel_save_path

    def add_figure_by_path(self, fig_path, alt_text=None, title="", describe=""):
        if self.flag_debug_mode:
            return "debugging"
        rtn = None
        for md_hd in self._md_hd:
            rtn = md_hd.add_figure_by_path(fig_path, alt_text, title, describe)
        return rtn

    def __call__(self, data: str or list, flag_stdout=False, flag_log=False, flag_md=False, flag_csv=False, hd=None):
        if hd is not None:
            hd(data)
        if not any([flag_log, flag_csv, flag_md, flag_stdout]):
            flag_stdout = True
        if flag_stdout:
            if "\r" in data:
                self.terminal_ctrl.to_stdout(data)
            else:
                self.terminal_ctrl.to_stdout(data+"\n")
        if flag_log:
            for callable_log_hd in self._log_hd:
                callable_log_hd.add_line(data)
        if flag_md:
            for callable_md_hd in self._md_hd:
                callable_md_hd.add_line(data)

        meta_interface = self
        class MetaOutput:
            def __init__(self, data_):
                self.interface = meta_interface
                self.data = data_

            def __str__(self):
                return "" + self.data

            def __repr__(self):
                return f"MetaOutput[{str(self)}]"

            def md(self, data_=None):
                data_ = self.data if data_ is None else data_
                return self.interface(data_, flag_md=True)

            def log(self, data_=None):
                data_ = self.data if data_ is None else data_
                return self.interface(data_, flag_log=True)

            def stdout(self, data_=None):
                data_ = self.data if data_ is None else data_
                return self.interface(data_, flag_stdout=True)

        return MetaOutput(data)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        self.terminal_ctrl.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    def add_graph(self, model, input_to_model=None, verbose=False, use_strict_trace=True):
        return self.terminal_ctrl.add_graph(model, input_to_model, verbose, use_strict_trace)

    def get_editor_markdown(self):
        return self._md_hd

    def get_editor_log(self):
        return self._log_hd

    def rename_save_dir(self, new_save_dir_name):
        """
        将已有的save_dir的最后一个目录更名
        :param new_save_dir_name:
        :return:
        """
        old_save_dir = self.report_path
        self.report_base_name = new_save_dir_name
        new_save_dir = self.report_path
        if new_save_dir != old_save_dir:
            for handle in self.handles.values():
                handle.rename_save_dir(new_save_dir)
            if self._is_main():
                for sub_reporter in self.sub_reporter.values():
                    if not sub_reporter._is_main():
                        sub_reporter.rename_save_dir(new_save_dir_name)
                os.rename(old_save_dir, new_save_dir)
        return self.report_path



class SubReporter(Reporter):
    def band(self, display_name:str, obj=None, flag_split_log=None):
        self("[!-] you are trying to band a obj to a sub_reporter！")
        return self.args.reporter.band(display_name, obj, flag_split_log)
        # raise NotImplementedError("you can't band an object to a sub_reporter.")

    def _is_main(self):
        return False

class TqdmStream(TextIOWrapper):
    to_stdout = True

    def __init__(self):
        buffer_io = os.open(os.devnull, os.O_WRONLY)
        buffer = os.fdopen(buffer_io, 'bw')
        super().__init__(buffer)

    def set2_stdout(self, to_stdout):
        self.to_stdout = to_stdout

    def write(self, buf: str):
        if self.to_stdout:
            return sys.stdout.write(buf)
        else:
            return
            # super(TqdmStream, self).write(buf)

class TerminalControl(object):
    """
    主要用来管理命令行 方便中途输入指令用
    因此必须是单例模式
    """
    _instance = None
    driving_pbar = None

    def __new__(cls, args=None):
        #单例模式
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # cls._instance = super(TrackerCore, cls)
        return cls._instance

    def __init__(self, args):
        self.args = args
        self.tqdm_stream_ = TqdmStream()
        self.args.stdout_enable_ = True
        # self.args.sys_stdout_ = sys.stdout
        self.cmd_mode = False
        self.buffer_stdout = ""
        self.buffer_cmd = []
        self.tensorboard_summary_writer = None
        self.tensorboard_path = None

    def get_tb_summary_writer(self):
        if self.tensorboard_summary_writer is not None:
            return self.tensorboard_summary_writer
        else:
            # 检查torch.utils.tensorboard模块是否存在
            if importlib.util.find_spec("torch.utils.tensorboard") is not None:
                # 如果存在，导入模块
                from torch.utils.tensorboard import SummaryWriter
                self.tensorboard_path = os.path.join(self.args.tb_root_dir, datetime.now().strftime("%b%d_%H-%M-%S") + "_" + socket.gethostname())
                self.tensorboard_summary_writer = SummaryWriter(log_dir=self.tensorboard_path)
            else:
                self.tensorboard_summary_writer = None
                self.tensorboard_path = "< tensorboard is not available. >"
                # 如果不存在，可以选择执行一些替代操作或抛出异常
                print("[-] torch.utils.tensorboard is not available.")
            return self.tensorboard_summary_writer

    def silence_sys_out(self, silence=True):
        if silence:
            self.tqdm_stream_.set2_stdout(False)
            # sys.stdout = self.devnull
            # self.args.sys_stdout_ = os.fdopen(self.devnull, 'w')
        else:
            # sys.stdout = self.original_stdout
            self.tqdm_stream_.set2_stdout(True)
            # self.args.sys_stdout_.close()
            # self.args.sys_stdout_ = self.original_stdout

    def check_input(self, enable=True):
        if enable:
            if select.select([sys.stdin], [], [], 0)[0]:
                user_input = sys.stdin.readline()
                if user_input == "\n":
                    self.args.stdout_enable_ = False
                    if not self.cmd_mode:
                        self.cmd_mode = True
                        print("\n[!]Command mode is enabled, please input your command:")
                        self.silence_sys_out(True)
                    else:
                        self.cmd_mode = True

                else:
                    if self.cmd_mode:
                        cmd = user_input.strip()
                        self.silence_sys_out(False)
                        print(f"[!] receive command: [{cmd}]")
                        # if self.driving_pbar is not None:
                        #     self.driving_pbar.disable = False
                        self.buffer_cmd.append(cmd)
                        self.args.stdout_enable_ = True
                        self.cmd_mode = False
                    else:
                        print("\n[!]Hit enter to start interactive mode")

    def get_one_cmd(self):
        if self.buffer_cmd:
            return self.buffer_cmd.pop(0)
        else:
            return None

    def get_all_cmd(self):
        if self.buffer_cmd:
            rtn = self.buffer_cmd.copy()
            self.buffer_cmd.clear()
            return rtn
        else:
            return []

    def to_stdout(self, data):
        # self.buffer_stdout += "\r" + original_data
        self.buffer_stdout += data
        if self.args.stdout_enable_:
            print(self.buffer_stdout, end="")
            self.buffer_stdout = ""
            return True
        else:
            return False

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        if self.tensorboard_summary_writer is not None:
            self.tensorboard_summary_writer.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)
        else:
            return None

    def add_graph(self, model, input_to_model=None, verbose=False, use_strict_trace=True):
        if self.tensorboard_summary_writer is not None:
            return self.tensorboard_summary_writer.add_graph(model, input_to_model, verbose, use_strict_trace)
        else:
            return None


class Handle(object):
    """
    handle 总是绑定了一个输出的接口-文件或者命令行
    """
    def __init__(self, args, save_dir, mapping_file_name, prefix="+"):
        self.args = args
        self.interface_file_name = mapping_file_name
        self.save_dir = save_dir
        self.save_path = os.path.join(save_dir, mapping_file_name)

        self.prefix = prefix
        self.flag_enable = True
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    @property
    def handle_id(self):
        return  re.search("#[0-9]*$",self.prefix).group()

    def output_enable(self,enable=True):
        self.flag_enable = enable

    def rename_save_dir(self, new_save_dir):
        self.save_dir = new_save_dir
        self.save_path = os.path.join(new_save_dir, self.interface_file_name)
    def close(self):
        pass

class PlainTextHandle(Handle):
    def __init__(self,args, save_dir, prefix, mapping_file_name="main_log.log", logging_mode=False):
        start_time_str = get_str_time(dateDiv="/", timeDiv=":", datetimeDiv=" ")
        self._word_buffer = "This is a log. " + f"InitTime: {start_time_str}\n"
        self.logging_mode = logging_mode
        super().__init__(args, save_dir, mapping_file_name, prefix)

    @property
    def log_prefix(self):
        if self.logging_mode:
            return f"[{self.prefix}] {get_str_time()}: "
        else:
            return ""

    def __call__(self, string):
        return self.add_line(string)

    def add_line(self, string, ending="\n"):
        if len(string.split("\n")) > 3:
            self._word_buffer += self.log_prefix + " \n" + string + ending
        else:
            self._word_buffer += self.log_prefix + string + ending
        if self.flag_enable:
            with open(self.save_path, "a+", encoding="utf-8") as F:
                payload = re.sub("\n", " \n", self._word_buffer)
                F.write(payload)
            self._word_buffer = ""
        return self

class CSVHandle(PlainTextHandle):
    csv_head = ["project name", "time","epoch", "train_loss", "val_loss", "learning_rate", "time_elapsed(s)"]
    def __init__(self, args, save_dir, prefix, mapping_file_name="main_log.csv"):
        super().__init__(args, save_dir, prefix, mapping_file_name, False)
        self._word_buffer = ",".join(self.csv_head) + "\n"

    def __call__(self, string):
        return self.add_line(string)

    def add_raw(self, data):
        raw = ",".join([str(i) for i in data]) + "\n"
        self._word_buffer += raw
        if self.flag_enable:
            try:
                with open(self.save_path, "a+", encoding="utf-8") as F:
                    # payload = re.sub("\n", " \n", self._word_buffer)
                    payload = self._word_buffer
                    F.write(payload)
                self._word_buffer = ""
            except PermissionError:
                pass
        return self

    def add_line(self, string, ending="\n" ):

        # if len(string.split("\n")) > 3:
        #     self._word_buffer += self.log_prefix + " \n" + string + ending
        # else:
        #     self._word_buffer += self.log_prefix + string + ending
        # if self.flag_enable:
        #     with open(self.save_path, "a+", encoding="utf-8") as F:
        #         payload = re.sub("\n", " \n", self._word_buffer)
        #         F.write(payload)
        #     self._word_buffer = ""
        return self.add_raw(string)

class MarkdownHandle(PlainTextHandle):
    """
    相比于文字主要是图文结合
    """
    backup_type=["svg", "pdf", "png"]
    def __init__(self, args, save_dir,prefix, mapping_file_name="main_log.md"):
        super().__init__(args, save_dir, prefix, mapping_file_name, False)
        self.pic_source_dir = os.path.join(save_dir, "pics")
        if not os.path.exists(self.pic_source_dir):
            os.makedirs(self.pic_source_dir)

        self.backup_dir = os.path.join(self.pic_source_dir, "other_type")
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
        # self._fig_count = 0
        self.used_name_count_dict = {"fig": 1}

    def _backup_fig_other_type(self,fig_data, save_name, used_ext=()):
        if "gif" in used_ext:
            return
        dir_name, file_name = os.path.split(save_name)
        if dir_name != "":
            if not os.path.exists(os.path.join(self.backup_dir, dir_name)):
                os.makedirs(os.path.join(self.backup_dir, dir_name))

        for fmt in self.backup_type:
            if fmt not in used_ext:
                fig_data.savefig(os.path.join(self.backup_dir, save_name + "." + fmt),
                                 format=fmt, bbox_inches='tight', transparent=True)

    def _fig_handle(self, fig_data, save_name, ext_):
        if save_name in self.used_name_count_dict:
            self.used_name_count_dict[save_name] += 1
            save_name = save_name + "_" + str(self.used_name_count_dict[save_name])
        else:
            self.used_name_count_dict.update({save_name: 1})
        if ext_ is None:
            ext = "png"
        else:
            ext = ext_
        fig_save_path = os.path.join(self.pic_source_dir, save_name + "." + ext)
        if not os.path.exists(os.path.dirname(fig_save_path)):
            os.makedirs(os.path.dirname(fig_save_path))
        # save_path = fig_data.savefig(fig_save_path, format=ext, bbox_inches='tight', transparent=True)
        save_path = fig_data.savefig(fig_save_path, format=ext, bbox_inches=None, transparent=True)
        if ext_ is None:
            self._backup_fig_other_type(fig_data, save_name,[ext])
        if save_path is None:
            save_path = fig_save_path
        rel_save_path = os.path.relpath(save_path, os.path.dirname(self.save_path))
        return rel_save_path

    def save_figure_only(self, fig_data, save_name="fig.svg"):
        name, ext = os.path.splitext(save_name)
        rel_save_path = self._fig_handle(fig_data, name, ext[1:] if len(ext) > 3 else None)
        return rel_save_path

    def add_figure_by_path(self, rel_save_path, alt_text=None, title="", describe=""):
        alt_text = alt_text if alt_text is not None else os.path.basename(rel_save_path)
        self.add_line(title)
        gram = f"![{alt_text}]({rel_save_path})"
        self.add_line(gram)
        self.add_line(describe)
        return rel_save_path

    def add_figure(self, fig_data, save_name="fig", alt_text=None, title="", describe=""):
        rel_save_path = self.save_figure_only(fig_data, save_name)
        return self.add_figure_by_path(rel_save_path, alt_text, title, describe)

    def rename_save_dir(self, new_save_dir):
        super().rename_save_dir(new_save_dir)
        self.pic_source_dir = os.path.join(new_save_dir, "pics")
        self.backup_dir = os.path.join(self.pic_source_dir, "other_type")

class TensorBoardHandle(Handle):
    pass
