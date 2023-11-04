# FDTR

**FDTR: Focus, Detect, Track and Re-Focus.** A Python based implementation of KTB2019 Radar Contest.

**Update:** data are now availabel via http://csdata.org/en/p/386/ , note that the order of data files changed.

本项目是第二届空天杯竞赛雷达组实现代码，下载之后，将数据放在`data/`目录下，然后参照`run.sh`脚本（注意这个脚本中文件目录是绝对目录，请根据自己实际情况修改或改为相对目录即可）以及`main.py`的接口，即可复现RD Focus组竞赛成绩。

如果觉得本代码对你有帮助，不要忘了给这个项目一个star。若你论文中使用了本代码或思路，请引用我们的文章：
```bibtex
@article{liu2019fdtr,
  title={聚焦、检测、跟踪、再聚焦：基于经典跟踪前检测框架的实时小目标检测与增强技术},
  author={刘本源，宋志勇，范红旗},
  journal={航空兵器},
  volume={26},
  number={6},
  eid={1},
  pages={6},
  year={2019},
  publisher={航空兵器}
}
```

