#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :flow_gif.py
# @Time      :2024/1/2 20:57
# @Author    :Oliver
import matplotlib.pyplot as plt
# import torch, gif
import matplotlib.animation as animation
import numpy as np
import tqdm
# from ...Tracker.Reporter import Report
from ...painter_format import PlotTemplate


# _glob_pbar: tqdm.tqdm = tqdm.tqdm()
class _MyFrame:
    def __init__(self, frameData, figure, pbar=None):
        self.frame = frameData
        self.figure = figure
        self.pbar = pbar

    def savefig(self, FigPath, *args, **kwargs):
        # global _glob_pbar
        if len(FigPath) < 4 or FigPath[-4] != '.':
            FigPath += ".gif"
        else:
            FigPath = FigPath[:-4] + ".gif"
        if self.pbar is not None:
            self.pbar.set_description(f"saving frame")
        self.frame.save(FigPath, writer='pillow')
        if self.pbar is not None:
            self.pbar.set_description(f"frame saved")
            self.pbar.close()
        # print("saved !")
        # ani = animation.ArtistAnimation(self.fig, self.frame, interval=50, blit=True,
        #                                 repeat_delay=1000)ni
        # self.frame[0].save(FigPath, save_all=True, loop=True, append_images=self.frame[1:],
        #                    duration=20, disposal=2)
        return FigPath


class BoundManager:
    _shrink_scalar = 0.98  # 在坐标轴相比上一帧缩小时减少的比例量
    _zoom_scalar = 1.01  # 在坐标轴相比上一帧增大时（但未超界）时增加的比例量
    _bound_factor = 1.10  # 坐标轴会预留的容限

    def __init__(self, para_num):
        self._para_num = para_num
        self.last_bound = [0] * para_num
        self.last_bound_draw = [0] * para_num

    def update(self, *args):
        rtn = []
        for i, para_now in enumerate(args):
            para_last = self.last_bound[i]
            para_last_draw = self.last_bound_draw[i]
            if self._bound_factor * para_now > para_last_draw:  # 膨胀
                if self._bound_factor * para_now > self._bound_factor * para_last_draw:
                    para_rtn = self._bound_factor * para_now    # 突变
                else:
                    para_rtn = min(self._bound_factor * para_now, self._zoom_scalar * para_last_draw)
            else:   # 缩小
                para_rtn = max(self._bound_factor * para_now, self._shrink_scalar * para_last_draw)
            self.last_bound[i] = para_now
            self.last_bound_draw[i] = para_rtn
            rtn.append(para_rtn)
        return rtn



_shrink_scalar = 0.98  # 在坐标轴相比上一帧缩小时减少的比例量
_zoom_scalar = 1.01  # 在坐标轴相比上一帧增大时（但未超界）时增加的比例量
_bound_factor = 1.10  # 坐标轴会预留的容限


def flow_gif_animation(lookback, predictions: dict, label, raw_track, graph_idx=0, dim=(0, 1, 2), focus=False,
                       model_weight=None, stride=1, basic_interval_ms=50):
    dim = list(dim)[:3]

    known_offset = lookback.shape[1]
    in_len = known_offset
    out_len = label.shape[1]
    model_num = len(predictions)

    real_data = label.cpu().detach().numpy()
    input_data = lookback.cpu().detach().numpy()
    raw_track = raw_track.cpu().detach().numpy()

    weight = np.ones(model_num)
    if model_weight is not None:
        input_len = len(model_weight)
        if input_len < model_num:  # 输入的较小 只赋值一部分
            weight[:input_len] = model_weight
        else:  # 输入的较大 截取前半段
            weight = model_weight[:model_num]
        weight /= weight.max()
    # weight 总是在0～1之间且一定有1

    weight_alpha = weight * 0.8 + 0.2
    weight_line_width = weight + 0.5

    prefixes = []
    pred_data = []
    for k in predictions:
        prefixes.append(k)
        pred_data.append(predictions[k].cpu().detach().numpy())
    sort_args = np.argsort(weight)
    prefixes = np.array(prefixes)[sort_args]
    pred_data = np.array(pred_data, dtype=object)[sort_args]

    # 假设输出的长度都是一样的 对齐样本
    sample_num = min([i.shape[0] for i in pred_data])
    pred_data_align = [i[-sample_num:] for i in pred_data]
    output_data = np.stack(pred_data_align, -1)
    frame_len = sample_num
    frame_iter = range(0, frame_len, stride)

    # 初始化画布
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection="3d" if len(dim) == 3 else None)

    title = plt.title(f"idx:{graph_idx} step:{0}")
    if len(dim) == 3:
        ax.set_xlabel("Longitude(°)")
        ax.set_ylabel("Latitude(°)")
        ax.plot(raw_track[:, dim[0]], raw_track[:, dim[1]], raw_track[:, dim[2]], color="lightgray",
                linewidth=0.5)  # 作为参照
        LastExtend = BoundManager(1)
    elif len(dim) == 2:
        ax.set_xlabel("Longitude(°)")
        ax.set_ylabel("Latitude(°)")
        ax.set_aspect('equal')
        ax.plot(raw_track[:, dim[0]], raw_track[:, dim[1]], color="lightgray", linewidth=0.5)  # 作为参照
        LastExtend = BoundManager(2)
    else:
        ax.set_xlabel("Time Frame")
        ax.set_ylabel("")
        ax.plot(raw_track[:, dim[0]], color="lightgray", linewidth=0.5)  # 作为参照
        ax.set_ylim(min(raw_track[..., dim[0]].min(), output_data[dim[0]].min()),
                    max(raw_track[..., dim[0]].max(), output_data[dim[0]].max()))
        LastExtend = BoundManager(3)
    known_real_ani = ax.plot((), ())[0]
    label_ani = ax.plot((), (), linewidth=1.25, label="realdata")[0]
    used_ani = ax.plot((), (), linewidth=1.25, label="lookback")[0]
    # output_ani = ax.plot((), (), linewidth=1.5, label=[f"predict_{prefix}" for prefix in prefixes])[0]
    output_ani = [ax.plot((), (), linewidth=weight_line_width[i], alpha=weight_alpha[i], label=f"predict_{prefix}")[0]
                  for i, prefix in enumerate(prefixes)]
    plt.legend()

    pbar = tqdm.tqdm(total=frame_len, unit="frames")
    # last_x_extend = 0
    # last_y_extend = 0   # 相比于之前的xy，可以快速扩大但不能快速减少


    def update_3d(num: int):
        pbar.update(stride)
        title.set_text(f"idx:{graph_idx} step:{num}")
        known_real_ani.set_data(raw_track[:num + known_offset, dim[0]], raw_track[:num + known_offset, dim[1]])
        label_ani.set_data(real_data[num, :, dim[0]], real_data[num, :, dim[1]])
        used_ani.set_data(input_data[num, :, dim[0]], input_data[num, :, dim[1]])
        for i in range(model_num):
            output_ani[i].set_data(output_data[num, :, dim[0], i], output_data[num, :, dim[1], i])

        known_real_ani.set_3d_properties(raw_track[:num + known_offset, dim[2]])
        label_ani.set_3d_properties(real_data[num, :, dim[2]])
        used_ani.set_3d_properties(input_data[num, :, dim[2]])
        for i in range(model_num):
            output_ani[i].set_3d_properties(output_data[num, :, dim[2], i])
        if focus:
            xs = np.hstack([real_data[num, :, dim[0]],
                            input_data[num, :, dim[0]],
                            output_data[num, :, dim[0]].reshape(-1)])
            ys = np.hstack(
                [real_data[num, :, dim[1]], input_data[num, :, dim[1]], output_data[num, :, dim[1]].reshape(-1)])
            x_extend = (xs.max() - xs.min()) * 0.05
            y_extend = (ys.max() - ys.min()) * 0.05
            ax.set_xlim(xs.min() - x_extend, xs.max() + x_extend)
            ax.set_ylim(ys.min() - y_extend, ys.max() + y_extend)
        return [known_real_ani, label_ani, used_ani, *output_ani]

    def update_2d(num: int, last_extend=LastExtend):
        # if num % 32 == 4:
        #     print('\r Gif frame step {} / {}'.format(num, frame_len), end="")
        pbar.update(stride)
        title.set_text(f"idx:{graph_idx} step:{num}")
        known_real_ani.set_data(raw_track[:num + known_offset, dim[0]], raw_track[:num + known_offset, dim[1]])
        label_ani.set_data(real_data[num, :, dim[0]], real_data[num, :, dim[1]])
        used_ani.set_data(input_data[num, :, dim[0]], input_data[num, :, dim[1]])
        for i in range(model_num):
            output_ani[i].set_data(output_data[num, :, dim[0], i], output_data[num, :, dim[1], i])
        if focus:
            xs = np.hstack(
                [real_data[num, :, dim[0]], input_data[num, :, dim[0]], output_data[num, :, dim[0]].reshape(-1)])
            ys = np.hstack(
                [real_data[num, :, dim[1]], input_data[num, :, dim[1]], output_data[num, :, dim[1]].reshape(-1)])
            real_x = np.hstack([real_data[num, :, dim[0]], input_data[num, :, dim[0]]])
            real_y = np.hstack([real_data[num, :, dim[1]], input_data[num, :, dim[1]]])
            x_mid = (real_x.max() + real_x.min()) * 0.5
            y_mid = (real_y.max() + real_y.min()) * 0.5

            x_lim = max(xs.max() - x_mid, x_mid - xs.min())
            y_lim = max(ys.max() - y_mid, y_mid - ys.min())
            x_extend, y_extend = last_extend.update(x_lim, y_lim)

            ax.set_xlim(x_mid - x_extend, x_mid + x_extend)
            ax.set_ylim(y_mid - y_extend, y_mid + y_extend)
        return [known_real_ani, label_ani, used_ani, *output_ani]

    def update_1d(num: int, last_extend=LastExtend):
        # if num % 32 == 4:
        #     print('\r Gif frame step {} / {}'.format(num, frame_len), end="")
        pbar.update(stride)
        title.set_text(f"idx:{graph_idx} step:{num}")
        known_real_ani.set_data(np.arange(num + known_offset),
                                raw_track[:num + known_offset, dim[0]])
        label_ani.set_data(np.arange(num + in_len, num + in_len + out_len),
                           real_data[num, :, dim[0]])
        used_ani.set_data(np.arange(num, num + in_len),
                          input_data[num, :, dim[0]])
        for i in range(model_num):
            output_ani[i].set_data(np.arange(num + in_len, num + in_len + out_len),
                                   output_data[num, :, dim[0], i])

        ax.set_xlim(num, num + in_len + out_len)
        if focus:
            ys = np.hstack([real_data[num, :, dim[0]],
                            input_data[num, :, dim[0]],
                            output_data[num, :, dim[0]].reshape(-1)])
            real_y = np.hstack([real_data[num, :, dim[0]], input_data[num, :, dim[0]]])
            y_mid = (real_y.max() + real_y.min()) * 0.5
            y_lim = max(ys.max() - y_mid, y_mid - ys.min())
            y_extend, = last_extend.update(y_lim)
            # y_extend = (ys.max() - ys.min()) * 0.05
            ax.set_ylim(y_mid - y_extend, y_mid + y_extend)
        return [known_real_ani, label_ani, used_ani, *output_ani]

    if len(dim) == 3:
        ani = animation.FuncAnimation(fig=fig, func=update_3d, frames=frame_iter, interval=basic_interval_ms * stride,
                                      blit=True)
        return _MyFrame(ani, fig, pbar)
    if len(dim) == 2:
        ani = animation.FuncAnimation(fig=fig, func=update_2d, frames=frame_iter, interval=basic_interval_ms * stride,
                                      blit=True)
        return _MyFrame(ani, fig, pbar)
    if len(dim) == 1:
        ani = animation.FuncAnimation(fig=fig, func=update_1d, frames=frame_iter, interval=basic_interval_ms * stride,
                                      blit=True)
        return _MyFrame(ani, fig, pbar)

#
# def flow_gif_overall(sample, pred, label, raw_track, idx=0, projection=None):
#     frame_len = sample.shape[0]
#     known_offset = sample.shape[1]
#     flag_3d = True if projection == "3d" else False
#
#     raw_track = raw_track.detach().cpu().numpy()
#     label_data = label.cpu().detach().numpy()
#     used_data = sample.cpu().detach().numpy()
#     predict_data = pred.cpu().detach().numpy()
#
#     # 初始化画布
#     fig = plt.figure(figsize=(6, 4))
#     ax = plt.axes(projection=projection)
#     title = plt.title(f"idx:{idx} step:{0}")
#     ax.plot(*raw_track, color="lightgray", linewidth=0.5)  # 作为参照
#     known_real_ani = ax.plot((), (), label="Known real")[0]
#     label_ani = ax.plot((), (), linewidth=1.25, label="Label")[0]
#     used_ani = ax.plot((), (), linewidth=1.25, label="Used Measure")[0]
#     output_ani = ax.plot((), (), linewidth=1.5, label="Predict")[0]
#     plt.legend()
#
#     def update(num: int):
#         if num % 32 == 4:
#             print('\r Gif frame step {} / {}'.format(num, frame_len), end="")
#
#         title.set_text(f"idx:{idx} step:{num}")
#         known_real_ani.set_data(*raw_track[:2, :num + known_offset])
#         label_ani.set_data(*label_data[num, :2])
#         used_ani.set_data(*used_data[num, :2])
#         output_ani.set_data(*predict_data[num, :2])
#         if flag_3d:
#             known_real_ani.set_3d_properties(raw_track[2, :num + known_offset])
#             label_ani.set_3d_properties(label_data[num, 2])
#             used_ani.set_3d_properties(used_data[num, 2])
#             output_ani.set_3d_properties(predict_data[num, 2])
#         return known_real_ani, label_ani, used_ani, output_ani
#
#     # ani = animation.ArtistAnimation(fig, [update(i) for i in range(frame_len)], interval=20, repeat=True, blit=True)
#
#     ani = animation.FuncAnimation(fig=fig, func=update, frames=frame_len, interval=20, repeat=True, blit=True)
#     return _MyFrame(ani, fig)
#
#
# # def flow_try(sample, pred, label, raw_track, idx=0, projection=None):
# #     frame_len = sample.shape[0]
# #     known_offset = sample.shape[-1]
# #     flag_3d = True if projection == "3d" else False
# #
# #     raw_track = raw_track.detach().cpu().numpy()
# #     label_data = label.cpu().detach().numpy()
# #     used_data = sample.cpu().detach().numpy()
# #     predict_data = pred.cpu().detach().numpy()
# #
# #
# #
# #     # 初始化画布
# #     fig = plt.figure(figsize=(6, 4))
# #     gm = GifManager(fig)
# #     ax = plt.axes(projection=projection)
# #     title = plt.title(f"idx:{idx} step:{0}")
# #     ax.plot(*raw_track, color="lightgray", linewidth=0.5)  # 作为参照
# #     known_real_ani = ax.plot((), (), label="Known real")[0]
# #     label_ani = ax.plot((), (), linewidth=1.25, label="Label")[0]
# #     used_ani = ax.plot((), (), linewidth=1.25, label="Used Measure")[0]
# #     output_ani = ax.plot((), (), linewidth=1.5, label="Predict")[0]
# #     plt.legend()
# #
# #     gm.addAnimation(title,text_data=[f"idx:{idx} step:{num}" for num in range(frame_len)])
# #     gm.addAnimation(known_real_ani, xy_data=[raw_track[:2, :num + known_offset] for num in range(frame_len)],
# #                     z_data=[raw_track[2, :num + known_offset] for num in range(frame_len)] if flag_3d else None)
# #     gm.addAnimation(label_ani, xy_data=label_data[:, :2], z_data=label_data[:, 2] if flag_3d else None)
# #     gm.addAnimation(used_ani,xy_data=used_data[:, :2], z_data=used_data[:, 2] if flag_3d else None)
# #     gm.addAnimation(output_ani,xy_data=predict_data[:, :2], z_data=predict_data[:, 2] if flag_3d else None)
# #
# #     return gm.run(frame_len)
# #     # def update(num: int):
# #     #     if num % 32 == 4:
# #     #         print('\r Gif frame step {} / {}'.format(num, frame_len), end="")
# #     #
# #     #     title.set_text(f"idx:{idx} step:{num}")
# #     #     known_real_ani.set_data(*raw_track[:2, :num + known_offset])
# #     #     label_ani.set_data(*label_data[num, :2])
# #     #     used_ani.set_data(*used_data[num, :2])
# #     #     output_ani.set_data(*predict_data[num, :2])
# #     #     if flag_3d:
# #     #         known_real_ani.set_3d_properties(raw_track[2, :num + known_offset])
# #     #         label_ani.set_3d_properties(label_data[num, 2])
# #     #         used_ani.set_3d_properties(used_data[num, 2])
# #     #         output_ani.set_3d_properties(predict_data[num, 2])
# #     #     return known_real_ani, label_ani, used_ani, output_ani
# #     #
# #     # ani = animation.FuncAnimation(fig=fig, func=update, frames=frame_len, interval=20, repeat=True)
# #     # return _MyFrame(ani)
#
#
# def flow_gif_focus(sample, pred, label, raw_track, idx=0, projection=None):
#     frame_len = sample.shape[0]
#     known_offset = sample.shape[-1]
#     flag_3d = True if projection == "3d" else False
#
#     raw_track = raw_track.detach().cpu().numpy()
#     label_data = label.cpu().detach().numpy()
#     used_data = sample.cpu().detach().numpy()
#     predict_data = pred.cpu().detach().numpy()
#
#     # 初始化画布
#     fig = plt.figure()
#     # ax = fig.add_subplot(111, projection=projection)
#     ax = plt.axes(projection=projection)
#     title = plt.title(f"idx:{idx} step:{0}")
#     ax.plot(*raw_track, color="lightgray", linewidth=0.5)  # 作为参照
#     known_real_ani = ax.plot((), (), label="Known real")[0]
#     label_ani = ax.plot((), (), linewidth=1.25, label="Label")[0]
#     used_ani = ax.plot((), (), linewidth=1.25, label="Used Measure")[0]
#     output_ani = ax.plot((), (), ".", linewidth=1.5, label="Predict")[0]
#     # output_ani = ax.scatter((), (), label="Predict")
#     plt.legend()
#
#     def update(num: int):
#         if num % 32 == 4:
#             print('\r Gif frame step {} / {}'.format(num, frame_len), end="")
#
#         title.set_text(f"idx:{idx} step:{num}")
#         known_real_ani.set_data(*raw_track[:2, :num + known_offset])
#         label_ani.set_data(*label_data[num, :2])
#         used_ani.set_data(*used_data[num, :2])
#         output_ani.set_data(*predict_data[num, :2])
#         original_data = np.hstack([label_data[num, :2], used_data[num, :2], predict_data[num, :2]])
#         xs = original_data[0]
#         ys = original_data[1]
#         x_extend = (xs.max() - xs.min()) * 0.05
#         y_extend = (ys.max() - ys.min()) * 0.05
#         ax.set_xlim(xs.min() - x_extend, xs.max() + x_extend)
#         ax.set_ylim(ys.min() - y_extend, ys.max() + y_extend)
#         if flag_3d:
#             known_real_ani.set_3d_properties(raw_track[2, :num + known_offset])
#             label_ani.set_3d_properties(label_data[num, 2])
#             used_ani.set_3d_properties(used_data[num, 2])
#             output_ani.set_3d_properties(predict_data[num, 2])
#         return known_real_ani, label_ani, used_ani, output_ani
#
#     ani = animation.FuncAnimation(fig=fig, func=update, frames=frame_len, interval=20)
#     return _MyFrame(ani, fig)
#
#
# def flow_gif_single_line(sample, pred, label, raw_track, idx=0, show_dim=0):
#     frame_len = sample.shape[0]
#     in_len = known_offset = sample.shape[-1]
#     out_len = pred.shape[-1]
#
#     raw_track = raw_track.detach().cpu().numpy()
#     label_data = label.cpu().detach().numpy()
#     used_data = sample.cpu().detach().numpy()
#     predict_data = pred.cpu().detach().numpy()
#     frame_array = np.arange(frame_len + in_len + out_len + 1)
#     # 初始化画布
#     fig = plt.figure()
#
#     # ax = fig.add_subplot(111, projection=projection)
#     ax = plt.axes()
#     ax.set_xlabel('Frame Number')
#     ax.set_ylim(min(raw_track[show_dim].min(), predict_data[:, show_dim].min()),
#                 max(raw_track[show_dim].max(), predict_data[:, show_dim].max()))
#     title = plt.title(f"idx:{idx} step:{0}")
#     ax.plot(raw_track[show_dim], color="lightgray", linewidth=0.5)  # 作为参照
#     known_real_ani = ax.plot((), (), label="Known real")[0]
#     label_ani = ax.plot((), (), linewidth=1.25, label="Label")[0]
#     used_ani = ax.plot((), (), linewidth=1.25, label="Used Measure")[0]
#     output_ani = ax.plot((), (), ".", linewidth=1.5, label="Predict")[0]
#     # output_ani = ax.scatter((), (), label="Predict")
#     plt.legend()
#
#     def update(num: int):
#         if num % 32 == 4:
#             print('\r Gif frame step {} / {}'.format(num, frame_len), end="")
#
#         title.set_text(f"idx:{idx} step:{num}")
#         known_real_ani.set_data(frame_array[:num + known_offset], raw_track[show_dim, :num + known_offset])
#         label_ani.set_data(frame_array[num + in_len:num + in_len + out_len], label_data[num, show_dim])
#         used_ani.set_data(frame_array[num:num + in_len], used_data[num, show_dim])
#         output_ani.set_data(frame_array[num + in_len:num + in_len + out_len], predict_data[num, show_dim])
#         ax.set_xlim(frame_array[num], frame_array[num + in_len + out_len])
#         # ax.set_ylim(ys.min() - y_extend, ys.max() + y_extend)
#         return known_real_ani, label_ani, used_ani, output_ani
#
#     ani = animation.FuncAnimation(fig=fig, func=update, frames=frame_len, interval=20)
#     return _MyFrame(ani, fig)


# def gif_2d_flow(sample, pred, label, raw_track, idx=0, projection=None):
#     frame_len = sample.shape[0]
#     known_offset = sample.shape[-1]
#
#     # num = 0
#     raw_track = raw_track.detach().cpu().numpy()
#     label_data = label.detach().numpy()
#     used_data = sample.detach().numpy()
#     predict_data = pred.detach().numpy()
#
#     # 初始化画布
#     fig = plt.figure()
#     ax = plt.axes(projection=projection)
#     title = plt.title(f"idx:{idx} step:{0}")
#     ax.plot(*raw_track, color="lightgray", linewidth=0.5)  # 作为参照
#     known_real_ani = ax.plot((), (), (), label="Known real")[0]
#     label_ani = ax.plot((), (), (), linewidth=1.25, label="Label")[0]
#     used_ani = ax.plot((), (), (), linewidth=1.25, label="Used Measure")[0]
#     output_ani = ax.plot((), (), (), linewidth=1.5, label="Predict")[0]
#     ax.legend()
#
#     def update(num):
#         if num % 32 == 4:
#             print('\r Gif frame step {} / {}'.format(num ,frame_len),end="")
#         # Used_data = EstimateData[num]
#         # all_known_data = raw_track[:, :num+known_offset]
#
#         title.set_text(f"idx:{idx} step:{num}")
#         known_real_ani.set_data(*raw_track[:, :num+known_offset])
#         label_ani.set_data(*label_data[num])
#         used_ani.set_data(*used_data[num])
#         output_ani.set_data(*predict_data[num])
#         return known_real_ani,
#
#     ani = animation.FuncAnimation(fig=fig, func=update, frames=frame_len, interval=20)
#     return _MyFrame(ani)
#
# def gif_3d_flow(sample, pred, label, raw_track, idx=0):
#     frame_len = sample.shape[0]
#     known_offset = sample.shape[-1]
#     num = 0
#     # Used_data = EstimateData[25]
#     raw_track = raw_track.detach().cpu().numpy()
#     label_data = label.detach().numpy()
#     used_data = sample.detach().numpy()
#     predict_data = pred.detach().numpy()
#
#     # 初始化画布
#     fig = plt.figure()
#     ax = plt.axes(projection="3d")
#     title = plt.title(f"idx:{idx} step:{0}")
#
#
#     ax.plot(*raw_track, color="lightgray", linewidth=0.5)  # 作为参照
#     known_real_ani = ax.plot((), (), (), label="Known real")[0]
#     label_ani = ax.plot((), (), (), linewidth=1.25, label="Label")[0]
#     used_ani = ax.plot((), (), (), linewidth=1.25, label="Used Measure")[0]
#     output_ani = ax.plot((), (), (),  linewidth=1.5, label="Predict")[0]
#     ax.legend()
#
#     def set_3d_data(obj, x, y, z):
#         obj.set_data(x, y)
#         obj.set_3d_properties(z)
#
#     def update(num):
#         if num % 32 == 4:
#             print('\r Gif frame step {} / {}'.format(num ,frame_len),end="")
#         all_known_data = raw_track[:, :num+known_offset]
#         title.set_text(f"idx:{idx} step:{num}")
#
#         set_3d_data(known_real_ani, *all_known_data)
#         set_3d_data(label_ani, *label_data[num])
#         set_3d_data(used_ani, *used_data[num])
#         set_3d_data(output_ani, *predict_data[num])
#         return known_real_ani,
#
#     ani = animation.FuncAnimation(fig=fig, func=update, frames=frame_len, interval=20)
#     return _MyFrame(ani)
