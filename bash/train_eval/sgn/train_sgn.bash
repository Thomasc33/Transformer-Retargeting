#!/bin/bash

# Submit all training scripts via sbatch

# ar/ri cv first
sbatch train_ntu_sgn_ar_cview.bash
sbatch train_ntu_sgn_ri_cview.bash

sbatch train_ntu120_sgn_ri_cview.bash
sbatch train_ntu120_sgn_ar_cview.bash

sbatch train_etri_sgn_ar_cview.bash
sbatch train_etri_sgn_ri_cview.bash

# cset/csub
sbatch train_ntu_sgn_ar_csub.bash

sbatch train_ntu120_sgn_ar_cset.bash
sbatch train_ntu120_sgn_ar_csub.bash
sbatch train_ntu120_sgn_ri_cset.bash

sbatch train_etri_sgn_ar_csub.bash