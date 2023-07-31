# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone - Classifying Skin Diseases using Computer Vision

### Problem Statement
In Singapore, minor skin ailments are prevalent. One example would be eczema, which affects up to 20% of children and 10% of adults<sup>1</sup>. The top 4 skin diseases in Singapore are eczema, acne, viral and fungal skin infections<sup>2</sup>. Mild cases of all these conditions are self-treatable, but the general knowledge among the layman population regarding these conditions is often limited.  As a result, individuals frequently seek diagnosis and treatment at clinics, contributing to long waiting times and the necessity of taking medical leave from work. These factors contribute to increased healthcare burden and reduced productivity on both individual and societal levels.

The aim of this project is to develop an image classification model using computer vision techniques to accurately classify common skin diseases prevalent in Singapore, specifically acne, warts, eczema, psoriasis, and fungal skin infections. The model should assist individuals in obtaining a preliminary diagnosis of their skin conditions.

In addition to the image classification component, another goal is to deploy the model on a user-friendly platform that goes beyond disease classification. The platform will provide individuals with pertinent information regarding their diagnosed skin condition, including the disease course, prognosis, and available treatment options. It will also offer details about the availability, proper usage, potential side effects, and other relevant information about recommended medications for self-treatment.

By incorporating comprehensive information into the application/website, individuals will have access to a reliable and accessible resource that not only aids in diagnosing their skin conditions but also offers valuable insights into their specific disease, potential treatment paths, and medication-related details. This should allow them to explore self-treatment options with over-the-counter products and potentially reduce the need for clinic visits and associated waiting times. This comprehensive approach has the potential to not only alleviate the strain on clinics and reduce waiting times but also empower individuals to make informed decisions about their skin health, enhancing productivity at both the individual and societal levels.

<sup>1. https://www.a-star.edu.sg/News/astarNews/news/features/atopic-dermatitis-the-search-for-answers</sup>  
<sup>2. https://pubmed.ncbi.nlm.nih.gov/35535625/</sup>

### Objectives
-   Develop an image classification model using convolutional neural networks (CNNs) to accurately classify five skin diseases (acne, eczema, fungal skin infections, psoriasis and warts).
-   Create an user-friendly interface or application that allows individuals to upload images of their skin conditions. The application will provide a preliminary diagnosis along with helpful information and suggestions.

---

**Note: Model files are available in the releases section. Please download them and move them into the models directory if you want to run the code.**

---

### Data

**1. Kaggle (https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)**

- `Eczema (1677 images)`

- `Warts Molluscum and other Viral Infections (2103 images)`

- `Psoriasis pictures Lichen Planus and related diseases (2055 images)`

- `Tinea Ringworm Candidiasis and other Fungal Infections (1702 images)`

After downloading the raw folders, I removed microscopic images and extracted only the files with labels in the file name (e.g. 'eczema', 'warts', 'psoriasis', 'tinea'). Below are the results after filtering:

- `Eczema (875 images)`
- `Warts (504 images)`
- `Fungal Skin Infections (911 images)`
- `Psoriasis (664 images)`

**2. Dermnetnz.org (https://dermnetnz.org/image-library)**

- `311 acne images were scraped from 5 categories`

Categories include:
- Acne affecting the back images
- Acne affecting the face images
- Facial acne images
- Infantile acne images
- Steroid acne images

### Conclusion
**The model is accurate and can serve as a reliable tool for preliminary diagnosis of skin conditions**
-   The developed image classification model has demonstrated remarkable accuracy, achieving 99% accuracy on the training dataset and 90% accuracy on the testing dataset. It has also achieved >90% on other metrics such as f1-score, recall and precision.
-   This high level of accuracy indicates the model's effectiveness in accurately classifying common skin diseases.

**Fungal rashes, eczema and psoriasis are more difficult to differentiate**
-   For all models, the recall and f1-score is consistently the lowest for psoriasis. This is expected as psoriasis presents with a wide range of clinical features and has many different subtypes. For example, guttate psoriasis may be misclassified as acne, whereas thick scaly lesions in plaque psoriasis may be misclassified as eczema.
-   The higher recall and f1-score achieved for acne and warts in the skin disease classification model suggest that these conditions are relatively easier to classify accurately. One plausible explanation for this observation is that acne and warts are often more localized in nature compared to other skin conditions. For instance, in the case of acne, it commonly appears as individual pimples or lesions, occurring one at a time or in small clusters. This localized nature may result in distinct visual patterns that are easier for the model to recognize and classify correctly. Similarly, warts usually tend to be isolated rather than occurring in large clusters. This characteristic makes it easier for the model to distinguish warts from other skin conditions that may present with a more diffuse or widespread appearance.
-   On the other hand, skin conditions like eczema may manifest as broader and more extensive areas of inflammation and irritation, often with multiple patches or lesions close together. The complexity and variation in presentation for such conditions might pose a greater challenge for the model's classification performance.

In conclusion, the project successfully developed an accurate image classification model and implemented it in a user-friendly application/website. This comprehensive solution provides individuals with the means to obtain reliable preliminary diagnoses, access pertinent information about their skin conditions, and explore self-treatment options for common skin ailments. It has the potential to reduce the strain on clinics, minimize waiting times, and empower individuals to take control of their skin health. The project offers a valuable contribution towards improving healthcare accessibility and productivity in managing self-treatable skin conditions.

---

### Recommendations
**1. One-stop solution for self-treatable skin conditions**
-   The model can be incorporated into an application to serve as a one-stop solution for patients with self-treatable skin conditions.
-   The model serves as a reliable tool for individuals seeking a preliminary diagnosis for their skin conditions. By capturing and uploading images of their ailments, users can receive prompt and accurate identification of their skin disease.
-   The application can then provide pertinent information to the patient regarding his/her disease, including what a brief description of the condition, example images, disease course, current management, lifestyle tips, and medications for self-treatment.
-   The proposed solution will provide a holistic and seamless experience for patients, sparing them from the hassle of seeking information from multiple internet sources, encountering conflicting advice, and receiving varying medication recommendations. By integrating the model's reliable diagnosis and comprehensive information about their skin condition, patients can access all the necessary resources in one place, ensuring they receive accurate guidance on disease management, suitable medications, and where to obtain them.

**2. Integration with telemedicine and collaboration with dermatologists**
-   Integrating telemedicine features with our solution is a logical step that would truly benefit patients and our healthcare system.
-   Telemedicine can allow users to consult with dermatologists remotely. This feature can provide an additional layer of support by offering virtual consultations, advice, and guidance for more complex or non-self-treatable skin conditions like psoriasis. This would also be beneficial for patients who are less confident with the initial preliminary diagnosis or have not benefited from self-treatment after an arbitrary time period.
-   Collaboration with dermatology clinics or medical institutions can help facilitate this telemedicine aspect and provide users with a more holistic healthcare experience. Integration with medication delivery service would also be useful as this means patients can receive the entire healthcare package without stepping out of their houses.

**3. Expansion of dataset and continuous model improvement**
-   To enhance the model's generalization capabilities especially in the local context, it is recommended to expand the dataset by including a larger proportion of skin images originating from Singaporean patients.
-   Collaborating with healthcare professionals or institutions to gather diverse and representative data can help capture a comprehensive spectrum of skin conditions prevalent in Singapore. Additionally, considering variations in age, gender, and ethnicity within the dataset can improve the model's accuracy across different populations.
-   Image segmentation techniques can be explored to focus on targeted skin regions. This can potentially increase the model accuracy and enable the model to detect mixed skin lesions.
-   Multi-label, multi-output algorithms can be explored. For example, the model can be trained to detect specific subtypes of fungal skin infections, such as candidiasis, which requires different treatment.