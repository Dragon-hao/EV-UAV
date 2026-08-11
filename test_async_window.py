import torch
import tqdm

from configs.configs import cfg
from dataset.ev_uav_async import EvUAVAsync
from model.evspsegnet import evspsegnet
from utils.eval import evalute
import numpy as np

windows=[
    100,
    200,
    500,
    1000,
    2000,
    4000,
    8000
]


if __name__ == '__main__':

    device = "cuda:0"


    # ==========================
    # 模型只加载一次
    # ==========================

    net = evspsegnet(cfg).eval()
    net.cuda()


    net.load_state_dict(
        torch.load(
            cfg.model_path,
            map_location="cuda"
        )
    )

    print(
        "dict load:",
        cfg.model_path
    )


    # ==========================
    # 不同时间窗口实验
    # ==========================

    for window in windows:

        print("\n====================")
        print(
            "window:",
            window,
            "ms"
        )
        print("====================")


        # async dataset
        dataset = EvUAVAsync(
            cfg,
            mode='test',
            window_ms=window,
            stride_ms=window
        )


        print(
            "window:",
            window,
            "samples:",
            len(dataset)
        )


        test_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=dataset.custom_collate
        )


        evaluator = evalute(cfg)


        pbar = tqdm.tqdm(
            total=len(test_dataloader),
            desc=f'{window}ms',
            unit='video',
            unit_scale=True
        )


        event_nums = []


        for sample, ev in enumerate(test_dataloader):
            if sample < 3:
                print(
                    "idx_label:",
                    type(ev['idx_label']),
                    np.shape(ev['idx_label'])
                )

            with torch.no_grad():

                x = ev['voxel_ev']


                label = (
                    ev['seg_label']
                    .float()
                    .cuda()
                    .reshape(-1)
                )


                p2v_map = (
                    ev['p2v_map']
                    .long()
                    .cuda()
                )


                ev_locs = (
                    ev['locs']
                    .float()
                    .cuda()
                )


                idx = ev['idx_label']

                # 保证idx始终是一维
                if hasattr(idx, "reshape"):
                    idx = idx.reshape(-1)


                # 统计事件数量
                event_nums.append(
                    len(label)
                )


                ts = ev_locs[:,3]


                preds, voxel = net(x)


                # voxel预测恢复到event级别
                preds = (
                    preds[p2v_map]
                    .reshape(-1)
                    .cpu()
                )


                if cfg.eval:

                    evaluator.matches[str(sample)] = {}

                    evaluator.matches[str(sample)]['seg_pred'] = preds

                    evaluator.matches[str(sample)]['seg_gt'] = label


                    if cfg.roc:

                        evaluator.roc_update(
                            ts.detach().cpu().reshape(-1),
                            preds.detach().cpu().reshape(-1),
                            idx,
                            label.detach().cpu().reshape(-1),
                            ev_locs.detach().cpu()
                        )


            pbar.update(1)


        pbar.close()


        # ======================
        # 计算指标
        # ======================

        if cfg.eval:

            iou = (
                evaluator
                .evaluate_semantic_segmantation_miou()
            )


            seg_acc = (
                evaluator
                .evaluate_semantic_segmantation_accuracy()
            )


            if cfg.roc:
                print(
                    "ROC stats:",
                    "obj_num=", evaluator.obj_num,
                    "correct_num=", evaluator.correct_num,
                    "frame_num=", evaluator.frame_num,
                    "false_num=", evaluator.false_num
                )
                pd, fa = evaluator.cal_roc()


                print(
                    f"window={window}ms "
                    f"IoU={iou:.4f} "
                    f"ACC={seg_acc:.4f} "
                    f"Pd={pd:.4f} "
                    f"Fa={fa:.8f} "
                    f"avg_events={sum(event_nums)/len(event_nums):.1f}"
                )

            else:

                print(
                    f"window={window}ms "
                    f"IoU={iou:.4f} "
                    f"ACC={seg_acc:.4f}"
                )