# IMAGE-CLASSIFICATION-MODEL



COMPANY_: CODTECH IT SOLUTIONS


_NAME_: VAISHNAVI VILAS BANSUDE


_INTERN ID_: CTIS4923


_DOMAIN_: MACHINE LEARNING


_DURATION_: 4 WEEKS


_MENTOR_: NEELA SANTOSH KUMAR

DESCRIPTION OF TASK:

Title: Brain Tumor MRI Image Classification Using Convolutional Neural Network (CNN)

Introduction:

The objective of this project is to build a Convolutional Neural Network (CNN) model for brain tumor classification using MRI images. Medical image analysis plays a significant role in assisting doctors in diagnosing serious diseases such as brain tumors. Manual diagnosis can be time-consuming and dependent on expert interpretation. Therefore, deep learning techniques like CNNs can help automate and improve the accuracy of tumor detection. In this project, a CNN model is developed using TensorFlow and Keras to classify MRI images into four categories: Glioma, Meningioma, Pituitary tumor, and No tumor.

Dataset Description:

The dataset used for this project is obtained from Kaggle and contains approximately 7,200 human brain MRI images. The dataset is divided into two main folders: Training and Testing. Each folder contains four subfolders corresponding to the tumor categories. The dataset is balanced, meaning each class contains a similar number of images. This balanced distribution helps in reducing bias during model training and ensures fair classification performance across all classes.

Data Preprocessing:

Before training the model, image preprocessing is performed using ImageDataGenerator. All images are resized to 150×150 pixels to maintain uniform input size. Pixel values are normalized from the range 0–255 to 0–1 to improve training efficiency. Data augmentation techniques such as rotation, zooming, and horizontal flipping are applied to the training dataset to increase data variability and reduce overfitting. Additionally, 20% of the training data is reserved for validation to monitor model performance during training.

CNN Model Architecture:

The CNN model consists of multiple convolutional and max-pooling layers. The convolutional layers extract important features such as edges and textures from MRI images. The number of filters increases progressively (32, 64, 128, and 256) to capture deeper features. Max-pooling layers reduce spatial dimensions and computational complexity.

After feature extraction, a flatten layer converts the feature maps into a one-dimensional vector. A fully connected dense layer with ReLU activation is added for classification learning. A dropout layer is included to prevent overfitting by randomly disabling neurons during training. The final output layer contains four neurons with softmax activation to classify the images into the four tumor categories.

Model Training and Evaluation:

The model is compiled using the Adam optimizer and categorical cross-entropy loss function, which is suitable for multi-class classification. The model is trained for 25 epochs using the training dataset, while validation accuracy and loss are monitored to prevent overfitting.

After training, the model is evaluated on the test dataset. Performance metrics include test accuracy, classification report (precision, recall, and F1-score), and confusion matrix. The model achieved satisfactory accuracy, demonstrating the effectiveness of CNNs in medical image classification tasks.

Conclusion:

This project successfully demonstrates the implementation of a CNN-based brain tumor classification system using MRI images. The model effectively learns complex image features and classifies tumors into four categories. The results highlight the potential of deep learning in assisting medical diagnosis. Future improvements may include the use of transfer learning and advanced architectures to further enhance performance and real-world applicability.

OUTPUT

<img width="572" height="455" alt="Image" src="https://github.com/user-attachments/assets/3dc3aca1-50d1-43cb-93ed-66a1a35da769" />

<img width="572" height="455" alt="Image" src="https://github.com/user-attachments/assets/fddec6c1-10ca-4f5e-ba3c-57df1274edd2" />

<img width="753" height="301" alt="Image" src="https://github.com/user-attachments/assets/4116fca3-53bd-4108-a58d-bfe691c1e381" />

<img width="539" height="438" alt="Image" src="https://github.com/user-attachments/assets/c730c685-fad6-447a-9b11-2895b3fda8b7" />

