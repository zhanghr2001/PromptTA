import os.path as osp
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class MY_DomainNet(DatasetBase):
    """DomainNet.

    Statistics:
        - Around 580k images.
        - 345 classes.
        - 6 domains: clipart, infograph, painting, quickdraw, real, sketch.
        - URL: [
                    "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
                    "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
                    "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
                    "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
                    "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
                    "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"
                ].

    """

    dataset_dir = "domainnet"
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

    def __init__(self, cfg):
        self.root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")
        self.split_dir = osp.join(self.dataset_dir, "splits")

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS)

        train = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "train")
        val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "test")
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, "all")

        super().__init__(train_x=train, val=val, test=test)

    def _read_data(self, input_domains, split):
        items = []

        for domain, dname in enumerate(input_domains):
            if split == "all":
                file_train = osp.join(
                    self.split_dir, dname + "_train.txt"
                )
                impath_label_list = self._read_split_domainnet(file_train)
                file_val = osp.join(
                    self.split_dir, dname + "_test.txt"
                )
                impath_label_list += self._read_split_domainnet(file_val)
            else:
                file = osp.join(
                    self.split_dir, dname + "_" + split + ".txt"
                )
                impath_label_list = self._read_split_domainnet(file)

            for impath, label in impath_label_list:
                classname = impath.split("/")[-2]
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=classname
                )
                items.append(item)

        return items
    
    def _read_split_domainnet(self, split_file):
        impath_label_list = []

        with open(split_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                impath = osp.join(self.image_dir, impath)
                label = int(label)  # start from 0
                impath_label_list.append((impath, label))

        return impath_label_list
