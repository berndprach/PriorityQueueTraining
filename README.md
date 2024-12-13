
# Priority Queue Training
Code for blog post [HardMiningForRobustClassification](https://berndprach.github.io/blog-posts/2024/12/HardMiningForRobustClassification/).
This repository contains the code for the [HighestLossQueue](src/training_queues/highest_loss_queue.py) and further the code to train 1-Lipschitz models.

## Usage:

First (optionally) do a learning rate search:
```bash
python run.py scripts\train\queue_model.py AOL-MLP 64 --alpha=0.1 --learning-rate-search
python run.py scripts\plot\lr_search_results.py AOL-MLP 64 --alpha=0.1

python run.py scripts\train\standard_model.py AOL-MLP 64 --learning-rate-search
python run.py scripts\plot\lr_search_results.py AOL-MLP 64 --alpha=-1
```

Train on full training set, evaluate on test set:
```bash
python run.py scripts\train\queue_model.py AOL-MLP 64 --alpha=0.1 --lr=0.14 --test
python run.py scripts\train\standard_model.py AOL-MLP 64 --lr=0.08 --test
```

Plot results:
```bash
python run.py scripts/plot/blog_plot.py AOL-MLP --radius 255
python run.py scripts/plot/blog_plot.py AOL-MLP --radius 36 --partition val
```


## License:
Copyright ©2024. Institute of Science and Technology Austria (ISTA). All Rights Reserved.  

This file is part of PriorityQueueTraining, which is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
 
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
 
You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
 
Contact the Technology Transfer Office, IST Austria, Am Campus 1, A-3400 Klosterneuburg, Austria, +43-(0)2243 9000, twist@ist.ac.at, for commercial licensing opportunities.
