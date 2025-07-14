# Bias Correction of Satellite Precipitation Estimation Using Deep Neural Networks

Satellite-based precipitation products (SPPs) have gained popularity among researchers due to their utility in hydrologic studies. Several gridded SPPs with global coverage, such as the Integrated Multi-satellite Retrievals for GPM (IMERG) and the PERSIANN family of products, are widely used. However, the accuracy of these products varies by retrieval algorithm and geographic location.

Numerous correction techniques have been developed, with machine learning—particularly deep neural networks—proving highly effective in improving precipitation estimation. This study investigates the performance of the PERSIANN-Dynamic Infrared Rain Rate near real-time product (PDIR-Now) over the Western U.S. and evaluates the effectiveness of three deep learning models: U-Net, Efficient-UNet, and a conditional Generative Adversarial Network (cGAN), in correcting biases.

The models integrate digital elevation data to account for orographic effects, which enhances accuracy beyond traditional correction techniques. Results show that Efficient-UNet and cGAN consistently outperform the original PDIR-Now and standard U-Net across various statistical and categorical metrics at multiple temporal scales. This bias correction framework holds promise for improving precipitation estimates and supporting water resource management in data-scarce regions.

---

## 🔑 Key Points

- **Efficient-UNet and cGAN models** significantly improve PDIR-Now precipitation estimates, outperforming both the original product and baseline U-Net in terms of accuracy and bias correction.

- **Daily-scale training** yields better results than sub-daily training, better capturing daily precipitation variability.

- **Incorporation of elevation data** enhances model performance, especially in complex terrains, although regional variation remains and may require further refinement.

---

## 🧠 Model Architectures

### U-Net
<img width="100%" alt="U-Net model" src="https://github.com/user-attachments/assets/d95b2592-a988-4530-a9c7-1ce93156f2cb" />

### Efficient-UNet
<img width="100%" alt="Efficient-UNet model" src="https://github.com/user-attachments/assets/839faeef-cd90-4456-9598-28292357a82b" />

### cGAN
<img width="100%" alt="cGAN model" src="https://github.com/user-attachments/assets/bd292c83-07d8-459a-ae88-4d888f7193ca" />

---

## 📊 Evaluation Example

<img width="100%" alt="Evaluation results" src="https://github.com/user-attachments/assets/83d022cf-3b8a-4a34-9d63-b1c04fe412ae" />

---

## 📌 Citation

Dao, V., Arellano, C. J., Nguyen, P., Almutlaq, F., Hsu, K., & Sorooshian, S. (2025). Bias Correction of Satellite Precipitation Estimation Using Deep Neural Networks and Topographic Information Over the Western U.S. *Journal of Geophysical Research: Atmospheres*, 130(4), e2024JD042181.

---

Let me know if you'd like help turning this into a `README.md` or formatting it for a GitHub Pages site!
