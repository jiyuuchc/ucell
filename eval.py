from absl import app, flags
from pathlib import Path
import numpy as np
import tifffile
import pandas as pd
from tqdm import tqdm
from ucell.metrics import LabelMetrics

flags.DEFINE_string("datadir", None, "test data dir")
flags.DEFINE_string("outputdir", "predictions", "logging dir")

def run(_):
    datapath = Path(flags.FLAGS.datadir)
    outpath = Path(flags.FLAGS.outputdir)

    results = []

    for label_fn in tqdm(datapath.glob("**/*_label.tif")):
        name = label_fn.name
        relative = label_fn.parent.relative_to(datapath)
        gt_label = tifffile.imread(label_fn)
        outpath_ = outpath / relative
        mask = tifffile.imread(outpath_/name.replace("_label", "_mask"))

        result = {'image': str(label_fn.relative_to(datapath))}

        label_metric = LabelMetrics()

        label_metric.update(mask, gt_label)

        result.update(label_metric.compute())

        results.append(result)
    
    df = pd.DataFrame.from_records(results)

    micro = df.drop("image", axis=1).mean()
    print(micro)

    df.to_csv(outpath/"report.csv", index=False)

if __name__ == "__main__":
    app.run(run)

