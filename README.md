# Bias-Correction-ML
Satellite-based precipitation products (SPPs) have gained popularity among researchers due to their utility in hydrologic studies. Several gridded satellite-based precipitation products with global coverage, such as the Integrated Multi-satellitE Retrievals for GPM (IMERG) and the Precipitation Estimation from Remotely Sensed Information using Artificial Neural Networks (PERSIANN) family of products, are available worldwide. However, the accuracy of these products may vary due to retrieval algorithms or geographic location. Numerous correction techniques have been implemented, and machine learning techniques, especially Deep Neural Networks, have proven to be the most effective in improving precipitation estimation. This study aims to investigate the performance of the PERSIANN-Dynamic Infrared Rain Rate near real-time product (PDIR-Now) in the Western U.S. and assess the effectiveness of three deep learning models including U-Net, Efficient-UNet, and a conditional Generative Adversarial Network (cGAN) in correcting biases present in the product. The developed models are expected to be more accurate than traditional methods, as they include digital elevation information and can resolve complex orographic enhancements in precipitation processes. This incorporation will mitigate the bias associated with SPPs, enabling further potential applications in water resource management. The findings revealed that the corrected results, utilizing the Efficient-UNet and cGAN models, surpassed the original PDIR-Now product and U-Net model across various statistical and categorical metrics at different temporal scales. This bias-correction scheme will enhance the assessment and understanding of precipitation patterns and can be used to improve the quality of precipitation estimates in other regions.

#Key Points

  Efficient-UNet and cGAN models improve PDIR-Now's precipitation estimates, outperforming the original data in accuracy and bias correction

  Daily data training yields better results than aggregated sub-daily data, capturing daily precipitation patterns more effectively

  Incorporating elevation improves model performance, but accuracy varies by region, highlighting the need for further refinement
  
<img width="2128" height="1444" alt="image" src="https://github.com/user-attachments/assets/83d022cf-3b8a-4a34-9d63-b1c04fe412ae" />

<img width="2128" height="949" alt="image" src="https://github.com/user-attachments/assets/d95b2592-a988-4530-a9c7-1ce93156f2cb" />

<img width="2128" height="1592" alt="image" src="https://github.com/user-attachments/assets/839faeef-cd90-4456-9598-28292357a82b" />

<img width="2128" height="1746" alt="image" src="https://github.com/user-attachments/assets/bd292c83-07d8-459a-ae88-4d888f7193ca" />
