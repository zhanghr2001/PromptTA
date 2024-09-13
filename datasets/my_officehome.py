import glob
import os.path as osp
from dassl.utils import listdir_nohidden
from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, Datum


@DATASET_REGISTRY.register()
class MY_OfficeHome(DatasetBase):
    """Office-Home.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """

    dataset_dir = "office_home_dg"
    domains = ["art", "clipart", "product", "real_world"]
    data_url = "https://drive.google.com/uc?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "office_home_dg.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train = self._read_data(
            self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS, "train"
        )
        val = self._read_data(
            self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS, "val"
        )
        test = self._read_data(
            self.dataset_dir, cfg.DATASET.TARGET_DOMAINS, "all"
        )

        super().__init__(train_x=train, val=val, test=test)

    def _read_data(self, dataset_dir, input_domains, split):

        def _load_data_from_directory(directory):
            folders = listdir_nohidden(directory)
            folders.sort()
            items_ = []

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(directory, folder, "*.jpg"))

                for impath in impaths:
                    items_.append((impath, label))

            return items_

        items = []

        for domain, dname in enumerate(input_domains):
            if split == "all":
                train_dir = osp.join(dataset_dir, 'images', dname, "train")
                impath_label_list = _load_data_from_directory(train_dir)
                val_dir = osp.join(dataset_dir, 'images', dname, "val")
                impath_label_list += _load_data_from_directory(val_dir)
            else:
                split_dir = osp.join(dataset_dir, 'images', dname, split)
                impath_label_list = _load_data_from_directory(split_dir)

            for impath, label in impath_label_list:
                class_name = impath.split("/")[-2].lower()
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=class_name
                )
                items.append(item)

        return items