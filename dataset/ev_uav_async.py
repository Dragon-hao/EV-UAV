import os
import numpy as np

from dataset.basedataset import BaseDataLoader


class EvUAVAsync(BaseDataLoader):

    def __init__(
        self,
        configs,
        mode='test',
        window_ms=10,
        stride_ms=None
    ):
    
        super().__init__(configs)
    
        self.mode = mode
    
        self.root = os.path.join(
            self.root,
            mode
        )
    
        self.file_list = sorted(
            os.listdir(self.root)
        )
    
    
        # EV-UAV timestamp already in ms
        self.window = window_ms
    
    
        if stride_ms is None:
            self.stride = window_ms
        else:
            self.stride = stride_ms
    
    
        self.samples=[]
    
    
        for file_id, filename in enumerate(self.file_list):
        
            data=np.load(
                os.path.join(
                    self.root,
                    filename
                )
            )
    
    
            ts=data['ev_loc'][:,2]
    
            start=ts.min()
            end=ts.max()
    
    
            current=start

            while current < end:
            
                window_end=min(
                    current+self.window,
                    end
                )
                self.samples.append(
                    (
                        file_id,
                        current,
                        window_end
                    )
                )
                current += self.stride
        print(
            "Async samples:",
            len(self.samples)
        )
    
    
    def __getitem__(self,index):

        file_id,start,end = self.samples[index]


        path=os.path.join(
            self.root,
            self.file_list[file_id]
        )


        events=np.load(path)



        ev_loc=events['ev_loc']


        evs_norm=events['evs_norm'][:,0:4]


        seg_label=events['evs_norm'][:,4]


        idx=events['evs_norm'][:,5]



        mask=(
            (ev_loc[:,2]>=start)
            &
            (ev_loc[:,2]<end)
        )



        ev_loc=ev_loc[mask]

        evs_norm=evs_norm[mask]

        seg_label=seg_label[mask]

        idx=idx[mask]



        # 空窗口跳过
        if len(ev_loc)==0:

            return self.__getitem__(
                (index+1)%len(self)
            )


        return {

            "ev_loc":
                ev_loc,

            "evs_norm":
                evs_norm,

            "seg_label":
                seg_label,

            "idx":
                idx

        }



    def __len__(self):

        return len(self.samples)