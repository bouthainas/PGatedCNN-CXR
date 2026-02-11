# Quantification of Lung Infection Severity in CXRs Using PGatedCNN-CXR

This repository contains the official implementation of a deep learning framework for automatic lung infection severity quantification from chest X-ray (CXR) images. The proposed method combines a parallel GatedCNN-based architecture with attention mechanisms and a segmentation-guided augmentation strategy to improve robustness and generalization across datasets.

Accurate assessment of infection severity is critical for clinical decision-making. Our model is trained on multiple public CXR datasets and benchmarked against state-of-the-art approaches, achieving superior performance in terms of Mean Absolute Error (MAE) and Pearson Correlation (PC). The segmented lung replacement augmentation further enhances adaptability to diverse lung conditions.

## Architecture

![ ](https://github.com/bouthainas/PGatedCNN-CXR/blob/main/Diagram.png)

The network adopts a parallel multi-branch design built from GatedCNN blocks and attention layers to capture complementary multi-scale representations. Feature fusion modules integrate global and local cues, followed by a regression head for continuous severity prediction.

Segmentation-Based Augmentation Strategy

![ ](https://github.com/bouthainas/PGatedCNN-CXR/blob/main/Augmentation.png)

We introduce a segmented lung replacement augmentation approach that uses anatomical masks to modify lung regions between samples while preserving global structure. This strategy increases data diversity, mitigates imbalance, and improves robustness to unseen pathologies.

# Team
## Core Contributors
* Dr. Bouthaina Slika, University of the Basque Country, Spain & Ho Chi Minh Open University, Vietnam bslika001@ikasle.ehu.eus
* Prof. Dr. Fadi Dornaika, IEEE member, Dept. of Artificial Intelligence, University of the Basque Country & IKERBAQUE foundation, Spain fadi.dornaika@ehu.es
* Prof. Dr. Karim Hammoudi, IEEE member, Group Imagery, Dept. of Computer Science, IRIMAS, Université de Haute-Alsace, France, karim.hammoudi@uha.fr

# Reference
> Karim Hammoudi, Halim Benhabiles, Mahmoud Melkemi, Fadi Dornaika, Ignacio Arganda-Carreras, Dominique Collard, and Arnaud Scherpereel. Deep learning on chest x-ray images to detect and evaluate pneumonia cases at the era of covid-19. Journal of medical systems, 45(7):1–10, 2021. https://doi.org/10.1007/s10916-021-01745-4
```
@article{Hammoudi2021,
author={Hammoudi, Karim
and Benhabiles, Halim
and Melkemi, Mahmoud
and Dornaika, Fadi
and Arganda-Carreras, Ignacio
and Collard, Dominique
and Scherpereel, Arnaud},
title={Deep Learning on Chest X-ray Images to Detect and Evaluate Pneumonia Cases at the Era of COVID-19},
journal={Journal of Medical Systems},
year={2021},
month={June},
day={08},
issn={1573-689X},
doi={10.1007/s10916-021-01745-4},
url={https://doi.org/10.1007/s10916-021-01745-4}
}
```
> Bouthaina Slika, Fadi Dornaika, Karim Hammoudi, and Vinh Truong Hoang. Automatic quantification of lung infection severity in chest x-ray images. In 2023 IEEE Statistical Signal Processing (SSP) Workshop, pages 418–422. IEEE, 2023. https://doi.org/10.1109/SSP53291.2023.10207986
```
@inproceedings{slika2023ssp,
title={Automatic Quantification of Lung Infection Severity in Chest X-ray Images},
author={Slika, Bouthaina
and Dornaika, Fadi
and Hammoudi, Karim
and Hoang, Vinh Truong},
booktitle={2023 IEEE Statistical Signal Processing (SSP) Workshop},
pages={418--422},
year={2023},
organization={IEEE},
doi={10.1109/SSP53291.2023.10207986},
url={https://doi.org/10.1109/SSP53291.2023.10207986}
}
```
> Bouthaina Slika, Fadi Dornaika, Hamid Merdji, and Karim Hammoudi.Lung pneumonia severity scoring in chest X-ray images using transformers. In Medical & Biological Engineering & Computing, pages 1-19. Springer, 2024.https://doi.org/10.1007/s11517-024-03066-3
```
@article{slika2024lung,
title={Lung pneumonia severity scoring in chest X-ray images using transformers},
author={Slika, Bouthaina
and Dornaika, Fadi and
Merdji, Hamid and
Hammoudi, Karim},
journal={Medical & Biological Engineering & Computing},
pages={1--19},
year={2024},
publisher={Springer},
doi= {10.1007/s11517-024-03066-3},
url={https://doi.org/10.1007/s11517-024-03066-3}
}
```
> Bouthaina Slika, Fadi Dornaika, and Karim Hammoudi. Multi-Score Prediction for Lung Infection Severity in Chest X-Ray Images. In IEEE Transactions on Emerging Topics in Computational Intelligence, pages 1-7. IEEE,2024.https://doi.org/10.1009/TETCI.2024.3359082
```
@article{slika2024multi,
author={Slika, Bouthaina
and Dornaika, Fadi 
and Hammoudi, Karim},
title={Multi-Score Prediction for Lung Infection Severity in Chest X-Ray Images},
journal={IEEE Transactions on Emerging Topics in Computational Intelligence},
pages={1--7},
year={2024},
month={January},
day={20},
publisher={IEEE},
doi={10.1009/TETCI.2024.3359082},
url={https://doi.org/10.1009/TETCI.2024.3359082}
}
```
![ ](https://github.com/bouthainas/PGatedCNN-CXR/blob/main/Affiliations.png)
