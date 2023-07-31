import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import keras
from PIL import Image

# Specify directories
model_path = Path(__file__).parent.parent / 'models/vgg_model.h5'
image_dir = Path(__file__).parent / 'images'

# Inverse mapping of classes
inv_map_classes = {0: 'Acne',
                   1: 'Eczema',
                   2: 'Fungal Skin Infection',
                   3: 'Psoriasis',
                   4: 'Warts'}

# Format page
st.set_page_config(page_title='Skin Condition Image Classifier',
                    page_icon=':adhesive_bandage:',
                    layout='wide',
                    initial_sidebar_state='expanded')

# Define functions
@st.cache_data
def preprocess(image_data, size, conv_array=False):
    img = Image.open(image_data)
    # Resize image
    resized_img = img.resize(size)
    if conv_array:
        # Convert image to numpy array, and add batch dimension
        img_array = tf.keras.preprocessing.image.img_to_array(resized_img)
        resized_img = tf.expand_dims(img_array, 0)
    
    return resized_img

@st.cache_resource
def load_model(file_path):
    return tf.keras.models.load_model(file_path)
model = load_model(model_path)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css(Path(__file__).parent / "style.css")

# Title
st.title("Skin Condition Predictor")

# Setting the top navigation bar
menu_bar = option_menu(menu_title = None,
                       options = ['Predictor', 'Condition', 'Management', 'Medications'],
                       icons = ['file-image','emoji-dizzy', 'bandaid', 'capsule-pill'],
                       default_index = 0, # which tab it should open when page is first loaded
                       orientation = 'horizontal',
                       styles={'nav-link-selected': {'background-color': '#FF7F7F'}})

# Setting the sidebar
sidebar = st.sidebar.selectbox('Select Condition', ['Acne', 'Eczema', 'Fungal Skin Infection', 'Psoriasis', 'Warts'], index=0)

# Predictor
if menu_bar == 'Predictor':
    st.header("Upload an image file for prediction")
    img = st.file_uploader(label="", type=["jpg", "png"])
    if img is not None:
        st.write(img.type)
        st.success("You have successfully uploaded an image")
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            st.image(img)

        # Preprocess the image and make a prediction
        preprocessed_img = preprocess(img, size=(224,224), conv_array=True)

        predict_proba = model.predict(preprocessed_img)
        sorted_proba = np.sort(predict_proba)

        first_index = np.where(predict_proba == sorted_proba[0, 4])[1][0]
        first_class = inv_map_classes[first_index]
        first_class_prob = predict_proba[0, first_index]

        second_index = np.where(predict_proba == sorted_proba[0, 3])[1][0]
        second_class = inv_map_classes[second_index]
        second_class_prob = predict_proba[0, second_index]

        third_index = np.where(predict_proba == sorted_proba[0, 2])[1][0]
        third_class = inv_map_classes[third_index]
        third_class_prob = predict_proba[0, third_index]

        # Image Analysis Results
        st.markdown('---')
        st.header("Image Analysis Results")
        st.markdown('---')

        # Example images for icons
        example_image_folder = os.path.join(image_dir, 'example')
        example_images = {'Acne': os.path.join(example_image_folder, 'acne_eg.jpeg'),
                          'Eczema': os.path.join(example_image_folder, 'eczema_eg.jpg'),
                          'Psoriasis': os.path.join(example_image_folder, 'psoriasis_eg.jpeg'),
                          'Fungal Skin Infection': os.path.join(example_image_folder, 'tinea_corporis_eg.jpeg'),
                          'Warts': os.path.join(example_image_folder, 'warts_eg.jpeg')}

        container_first = st.container()
        with container_first:
            col1, col2 = st.columns([3, 7])
            with col1:
                icon_image = preprocess(example_images[first_class], (500,500))
                st.image(icon_image)
                st.caption(f"<p style='text-align:center; font-size:16px;'>Example Image</p>", unsafe_allow_html=True)
            with col2:
                st.markdown(f'## {first_class}')
                st.markdown("#### Probability: {:.1f}%".format(first_class_prob*100))
                if first_class != 'Psoriasis':
                    st.markdown("#### Recommendations: Self-treatable")
                else:
                    st.markdown('#### Recommendations: Please consult a doctor')
                if first_class == 'Acne':
                    st.markdown("""
                        - If you have mild acne, you can try to treat yourself with nonpresription products.
                        - Please refer to the above tabs for more information on the condition, management and medications available.
                        - If you do not improve after 3 months of using nonprescription products, consult a healthcare provider for advice on the most effective treatments.
                        """)
                elif first_class == 'Eczema':
                    st.markdown("""
                        - If you have mild eczema, you can try to treat yourself with nonpresription products.
                        - Please refer to the above tabs for more information on the condition, management and medications available.
                        - If you do not improve after 2 weeks of using nonprescription products, consult a healthcare provider for advice on the most effective treatments.
                        """)
                elif first_class == 'Fungal Skin Infection':
                    st.markdown("""
                        - Fungal skin infections are generally self-treatable.
                        - Please refer to the above tabs for more information on the condition, management and medications available.
                        - If you do not improve after 2 weeks of using nonprescription products, consult a healthcare provider for advice on the most effective treatments.
                        """)
                elif first_class == 'Psoriasis':
                    st.markdown("""
                        - Psoriasis is potentially a life-long condition.
                        - Please consult a healthcare provider for professional advice.
                        - You may refer to the above tabs for more information on the condition and management available.
                        """)
                elif first_class == 'Warts':
                    st.markdown("""
                        - Warts are generally self-treatable.
                        - Exclusions for self-treatment include involvement of the face, nails, anus or genitalia.
                        - Please refer to the above tabs for more information on the condition, management and medications available.
                        - If you do not improve after 4 weeks of using nonprescription products, consult a healthcare provider for advice on the most effective treatments.
                        """)            

        st.markdown("---")
        with st.expander(label=second_class, expanded=False):
            col1, col2, col3 = st.columns([1, 2, 9])
            with col1:
                icon_image = preprocess(example_images[second_class], (100,100))
                st.image(icon_image)
            with col2:
                st.write("Probability: {:.1f}%".format(second_class_prob*100))

        st.markdown("---")
        with st.expander(label=third_class, expanded=False):
            col1, col2, col3 = st.columns([1, 2, 9])
            with col1:
                icon_image = preprocess(example_images[third_class], (100,100))
                st.image(icon_image)
            with col2:
                st.write("Probability: {:.1f}%".format(third_class_prob*100))

# Acne
if sidebar == 'Acne':
    # Acne Condition Tab
    if menu_bar == 'Condition':
        st.header('What is acne?')
        st.write('Acne (commonly known as pimples) is a skin condition that occurs when your hair follicles become plugged with oil and dead skin cells.')

        st.header('What causes acne?')
        st.markdown("""
            - Excess oil (sebum) production
            - Acne bacteria
            - Androgenic hormones like testosterone
            - Occlusion of hair follicles by oil and dead skin cells""")

        st.header('Symptoms')
        st.write('Acne may present with varying degrees of severity (least to most severe):')
        st.markdown("""
            - Whiteheads (closed plugged pores)
            - Blackheads (open plugged pores)
            - Small red, tender bumps (papules)
            - Pustules (pimples with pus at the tip)
            - Large, solid, painful lumps under the skin (nodules)
            - Painful, pus-filled lumps under the skin (cystic lesions)""")

        col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,1])
        col_list = [col1, col2, col3, col4, col5, col6]
        acne_image_folder = os.path.join(image_dir, 'acne')
        acne_images = {os.path.join(acne_image_folder, 'whitehead.png'): 'Whiteheads',
                       os.path.join(acne_image_folder, 'blackhead.png'): 'Blackheads',
                       os.path.join(acne_image_folder, 'papule.png'): 'Papule',
                       os.path.join(acne_image_folder, 'pustule.png'): 'Pustule',
                       os.path.join(acne_image_folder, 'cyst.png'): 'Cyst',
                       os.path.join(acne_image_folder, 'nodule.png'): 'Nodule'}

        for index, (image_path, caption) in enumerate(acne_images.items()):
            with col_list[index]:
                acne_image = preprocess(image_path, (100,100))
                st.image(acne_image, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>{caption}</p>", unsafe_allow_html=True)

        st.header('Where does it occur?')
        st.write('Acne usually appears on the face, but it also can appear on the chest, back, and shoulders.')

        st.header('What is the outlook for acne?')
        st.markdown("""
            - Acne tends to improve after the age of 25 years but may persist, especially in females
            - Acne often responds well to treatment, but responses may take 6 to 8 weeks, and acne may flare up from time to time
            - Scarring may occur if severe acne is not treated""")


    # Acne Management Tab
    elif menu_bar == 'Management':
        st.header('Pharmacological Therapies')
        st.subheader('Mild acne')
        st.markdown("""
            - Topical antimicrobials such as benzoyl peroxide
            - Topical antibiotics such as clindamycin lotion
            - Topical retinoids such as tretinoin/adapalene gel""")

        st.subheader('Moderate acne')
        st.markdown("""
            - Oral antibiotics such as tetracycline, erythromycin or minoxycycline
            - Oral antiandrogen therapy such as low-dose combined oral contraceptive pills
            - Oral isotretinoin may be used if acne is persistent or treatment-resistant""")

        st.subheader('Severe acne')
        st.markdown("""
            - Oral antibiotics at higher doses / longer durations
            - Oral isotretinoin""")
        st.markdown("---")
        

        st.header('Non-pharmacological Therapies')
        st.markdown("""
            - Laser and light-based therapies
            - Chemical peels
            - Frational microneedling radio frequency""")
        st.markdown("---")


        st.header("Tips for Managing Acne")
        st.markdown("""
            - Clean your skin gently with a mild soap (such as Cetaphil or QV wash)
            - Look for 'noncomedogenic' formulas for cosmetics and creams
            - Remove all dirt or make-up. Do not leave make-up overnight
            - Avoid scrubbing/touching your face
            - Shampoo your hair everyday, especially if it is oily
            - Do not squeeze, scratch, pick or rub your pimples. This can lead to skin infections, slower healing and scarring
            """)

    # Acne Medications Tab
    elif menu_bar == 'Medications':
        st.header("Over-the-counter Medicines")
        st.write("""These medicines can be bought off the shelf at retail pharmacies such as Guardian,
            Watsons, Unity, and certain hospitals.""")

        # Benzoyl Peroxide
        with st.expander(label="Benzoyl Peroxide", expanded=False):
            st.subheader("What is Benzoyl Peroxide?")
            st.markdown("""
                - Benzoyl peroxide is used in the treatment of non-inflammatory acne
                - It has antimicrobial and comedolytic properties""")

            st.subheader("Counselling Points")
            st.markdown("""
                - Apply once or twice daily to the affected areas only
                - Strengths available: 2.5%, 5%, 10%
                - Visible improvements typically occur within 3 weeks, with maximal effects in approimately 8-12 weeks""")
            benzoyl_se = {'Potential Side Effects': ['Dry skin', 'Skin peeling, redness, irritation'],
                    'Management': ['Apply moisturizer regularly', 'Start with a lower strength (2.5-5%) once daily, increase as tolerated only if not effective']}
            benzoyl_se_df = pd.DataFrame(benzoyl_se)
            benzoyl_se_df.index = benzoyl_se_df.index + 1
            st.dataframe(pd.DataFrame(benzoyl_se_df))

            st.subheader('Example Brands')
            col1, col2 = st.columns([1,1])
            with col1:
                benzac_img = Image.open((os.path.join(image_dir, 'acne', 'benzac.jpg')))
                st.image(benzac_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>Benzac Gel</p>", unsafe_allow_html=True)
            with col2:
                oxy5_img = Image.open((os.path.join(image_dir, 'acne', 'oxy5.jpg')))
                st.image(oxy5_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>Oxy 5 Lotion</p>", unsafe_allow_html=True)

        st.header("Pharmacy-only Medicines")
        st.markdown('These medicines can be bought from retail/hospital pharmacies **provided there are pharmacists available.**')
        with st.expander(label="Topical Retinoids", expanded=False):
            # Topical Retinoids
            st.markdown("**Note: You can only purchase this from a pharmacist if the patient is at least 12 years old.**")    
            st.subheader("What are Topical Retinoids?")
            st.markdown("""
                - Topical retinoids help to prevent and treat both non-inflammatory and inflammatory acne
                - They exert their effects by reducing inflammation, normalizing follicular hyperkeratosis and preventing
                formation of microcomedones
                - **Examples of topical retinoids include adapalene and tretinoin.** Note that tretinoin requires a prescription.
                """)

            st.subheader("Counselling Points")
            st.markdown("""
                - Wash the skin with cleanser and water, then pat dry.
                - Apply thinly once every night before sleeping to the **entire face or acne-prone regions**.
                - Acne may seem to get worse during the first two to three weeks. This is a normal reaction.
                - The best results may be achieved after four to seven weeks of treatment.""")
            retinoid_se = {'Potential Side Effects': ['Dry skin', 'Skin irritation'],
                    'Management': ['Apply moisturizer regularly', 'Start applying every other night initially. Once tolerated, increased to every night application']}
            retinoid_se_df = pd.DataFrame(retinoid_se)
            retinoid_se_df.index = retinoid_se_df.index + 1
            st.dataframe(pd.DataFrame(retinoid_se_df))

            st.subheader('Example Brands')
            col1, col2 = st.columns([1,1])
            with col1:
                adapalene_img = Image.open((os.path.join(image_dir, 'acne', 'adapalene.jpg')))
                st.image(adapalene_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>Differin Gel</p>", unsafe_allow_html=True)

        st.header("TLDR: Which medicines should I choose?")
        st.markdown("""
            - In general, adapalene is more effective than benzoyl peroxide, but benzoyl peroxide is more readily available.
            - If you only have a few pimples and you are not very bothered by it, benzoyl peroxide may be sufficient.
            - Otherwise, adapalene is always a good choice.""")

# Eczema
if sidebar == 'Eczema':

    # Eczema Condition Tab
    if menu_bar == 'Condition':
        st.header('What is eczema?')
        st.write("""Eczema (also known as atopic dermatitis) is a condition that causes your skin to become dry, itchy,
            and bumpy. It is the most common inflammatory skin condition worldwide. Eczema is long-lasting and 
            is characterised by frequent remission and relapse. It is not contagious.""")

        st.header('What causes eczema?')
        st.write("There is no single cause of eczema, and there are many theories regarding the underlying mechanisms.")
        st.write("Some of the more popular theories include:")
        st.markdown("""
            - Overactive immune system
            - Inherited abnormalities in the skin barrier
            - Skin microbiome imbalance
            - Environmental factors such as soap, chlorine, and smoke""")

        st.header('Symptoms')
        st.write("""Acute eczema is red, weeping, and may have blisters. Over time, the eczema becomes chronic and the 
            skin becomes less red, but instead thickened (lichenfied) and scaly. Cracking of the skin (fissures) can occur.""")
        st.write('Eczema symptoms can appear anywhere on the body, and vary widely from person to person. They may include:')
        st.markdown("""
            - Dry, cracked skin
            - Itchiness
            - Rash on swollen skin
            - Small, raised bumps
            - Oozing and crusting
            - Thickened skin
            - Darkening of skin
            - Raw, sensitive skin from scratching""")

        # Example images of acute/chronic eczema
        col1, col2 = st.columns([1,1])
        with col1:
            acute_eczema_img = Image.open(os.path.join(image_dir, 'eczema', 'acute_eczema.jpeg'))
            st.image(acute_eczema_img, use_column_width=True)
            st.caption("<p style='text-align:center; font-size:16px;'>Acute Eczema</p>", unsafe_allow_html=True)
        with col2:
            chronic_eczema_img = Image.open(os.path.join(image_dir, 'eczema', 'chronic_eczema.jpg'))
            st.image(chronic_eczema_img, use_column_width=True)
            st.caption("<p style='text-align:center; font-size:16px;'>Chronic Eczema</p>", unsafe_allow_html=True)

        st.header('What is the outlook for eczema?')
        st.markdown("""
            - Eczema affects up to 20\% of children and up to 10\% of adults in Singapore.
            - Sensitive skin persists lifelong. It is impossible to predict whether eczema will improve by itself 
            or not in an individual.
            - Children who developed eczema before the age of 2 years has a lower risk of persistent disease than 
            those who developed eczema later in childhood or adolescence.
            - Eczema is typically worst between the ages of two and four years, and often improves or even clears after this. However,
            atopic dermatitis may be aggravated or reappear in adult life due to exposure to irritants or allergens.""")


    # Eczema Management Tab
    elif menu_bar == 'Management':
        st.header('Pharmacological Therapies')
        st.subheader('Emollients and Moisturizers')
        st.markdown("""
            - Moisturizers are an essential component of treatment for eczema.
            - They need to be applied at least 2-3 times a day, even when there is no active eczema.
            - They can be applied liberally, even up to as frequently as every 1-2 hours in periods of active disease.
            - There are different types/brands of moisturizers. Choose one that is effective, comfortable and affordable.
            - Apply moisturizers immediately after bathing to prevent the skin from drying out""")

        st.subheader('Topical Steroids')
        st.markdown("""
            - Topical steroids are the mainstay treatment for mild-to-moderate atopic dermatitis.
            - There are different potencies of steroids available. A different potency may be required depending on the skin area and 
            disease severity.
            - They are safe and effective when used correctly.""")

        st.subheader('Topical Calcineurin Inhibitors')
        st.markdown("""
            - Topical calcineurin inhibitors (pimecrolimus, tacrolimus) are topical immunomodulators and work in a different way from steroids.
            - They are suitable for treating eczema in sensitive sites such as the eyelids, face, skin folds and genital areas.""")

        st.subheader('Oral Antihistamines')
        st.markdown("""
            - Oral antihistamines may be useful in helping control itch in eczema.
            - Sedating antihistamines may help with sleep disturbance that is common in eczema.""")

        st.subheader('Systemic Steroids')
        st.markdown("""
            - A short course of systemic corticosteroids may be useful to quickly control a flare and provide temporary control.
            - Prolonged usage of systemic steroids is discouraged as it has many side effects.""")
        st.markdown("---")
        

        st.header('Non-pharmacological Therapies')
        st.subheader('Phototherapy')
        st.markdown("""
            - Narrowband UVB phototherapy can be used to treat severe eczema.
            - Phototherapy is usually combined with the usual topical treatments.""")

        st.subheader('Wet Wraps')
        st.markdown("""
            - Wet wraps are useful for flares and recalcitrant eczema.
            - They may be useful for increasing the penetration of certain topical agents.
            - Only done if recommended by a dermatologist.""")
        st.markdown("---")


        st.header('Tips for Managing Eczema')
        st.markdown("""
            - Choose a moisturizer that is free of additives, perfumes and fragrances.
            - Avoid scratching the affected area.
            - Avoid sodium lauryl sulfate and strong detergents in cleansers.
            - Avoid carpets, rugs and soft toys at home. If unavoidable, wash them with hot water once every 2 weeks.
            - Avoid hot, frequent and long baths. Try to bathe with cool or lukewarm water, and limit your baths to 15 minutes and twice a day. 
            """)



    # Eczema Medications Tab
    elif menu_bar == 'Medications':
        # OTC
        st.header("Over-the-counter Medicines")
        st.write("""These medicines can be bought off the shelf at retail pharmacies such as Guardian,
            Watsons, Unity, and certain hospitals.""")
        # Topical Steroids (Least Potent)
        with st.expander(label="Topical Steroids (Least Potent)", expanded=False):
            st.subheader("What are topical steroids?")
            st.markdown("""
                - Topical steroids and moisturizers are the mainstay of therapy for mild to moderate eczema.
                - Topical steroids help with eczema due to their anti-inflammatory properties.
                - **Hydrocortisone 1\% cream is available off the shelf. It has the lowest potency amongst topical steroids.**""")

            st.subheader("Counselling Points")
            st.markdown("""
                - Apply a thin layer to the affected area, twice daily.
                - One fingertip unit (FTU) is enough to treat a skin area of two adult hand sizes.
                - Continue application until the skin is no longer red, itchy, or bumpy.
                - If the skin flares (i.e. becomes red, itchy, or bumpy), immediately start application again.
                - If you do not see any improvement in 2 weeks despite proper application, or experience pain,
                bleeding, fever or difficulty sleeping for a few nights, please contact a healthcare professional.  """)

            col1, col2 = st.columns([1,1])
            with col1:
                ftu_img = Image.open((os.path.join(image_dir, 'eczema', 'fingertip.jpg')))
                st.image(ftu_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>1 FTU can cover 2 adult hand areas</p>", unsafe_allow_html=True)

            st.subheader('Example brands')
            col1, col2 = st.columns([1,1])
            with col1:
                hydrocort_img = Image.open((os.path.join(image_dir, 'eczema', 'hydrocort.jpg')))
                st.image(hydrocort_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>Hydrocortisone cream</p>", unsafe_allow_html=True)

        # Oral Antihistamines
        with st.expander(label="Oral Antihistamines", expanded=False):
            st.subheader("What are oral antihistamines?")
            st.markdown("""
                - Oral antihistamines block the release of histamines, which is a potent inflammatory mediator commonly associated with allergic reactions.
                - They may be useful for temporary relief of itch in eczema.
                - **Examples include loratadine and cetirizine.**""")

            st.subheader("Counselling Points")
            st.markdown("""
                - (For Zyrtec-R/Clarityn) Take one tablet once a day for the relief of itch.""")

            st.subheader('Example brands')
            col1, col2 = st.columns([1,1])
            with col1:
                zyrtec_img = Image.open((os.path.join(image_dir, 'eczema', 'zyrtec.jpg')))
                st.image(zyrtec_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>Zyrtec-R (Cetirizine)</p>", unsafe_allow_html=True)
            with col2:
                clarityn_img = Image.open((os.path.join(image_dir, 'eczema', 'clarityn.jpg')))
                st.image(clarityn_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>Clarityn (Loratadine)</p>", unsafe_allow_html=True)

        # Pharmacy-only medicines
        st.header("Pharmacy-only Medicines")
        st.markdown('These medicines can be bought from retail/hospital pharmacies **provided there are pharmacists available.**')
        # Topical Steroids (Low-high potency)
        with st.expander(label="Topical Steroids (Low-high potency)", expanded=False):
            st.markdown("**Note: You can only purchase this from a pharmacist if the patient is at least 18 years old.**")    
            st.subheader("What are topical steroids?")
            st.markdown("""
                - Topical steroids and moisturizers are the mainstay of therapy for mild to moderate eczema.
                - Topical steroids help with eczema due to their anti-inflammatory properties.
                - Different steroid ingredient, strengths, and formulations contribute to their potencies.
                Refer to the table below for more information on steroid potencies.
                - **Topical steroids available as pharmacy medications include: Betamethasone, Desonide, Mometasone, Triamcinolone.**
                """)
            steroid_potency = {'Potency Group': ['Super-high potency (Group 1)',
                                             'High potency (Group 2)',
                                             'High potency (Group 3)',
                                             'Medium Potency (Group 4)',
                                             'Lower-mid potency (Group 5)',
                                             'Low potency (Group 6)',
                                             'Least potent (Group 7)'],
                           'Corticosteroid': ['Clobetasol Propionate 0.05% cream/ointment',
                                              'Betamethasone dipropionate 0.05% ointment',
                                              'Mometasone furoate 0.1% ointment',
                                              'Mometasone furoate 0.1% cream',
                                              'Betamethasone valerate 0.1% cream',
                                              'Desonide 0.05% cream',
                                              'Hydrocortisone 1% cream']}

            steroid_potency_df = pd.DataFrame(steroid_potency)
            steroid_potency_df.index = steroid_potency_df.index + 1
            st.dataframe(pd.DataFrame(steroid_potency_df))

            st.subheader("Counselling Points")
            st.markdown("""
                - Apply a thin layer to the affected area, two times a day.
                - One fingertip unit (FTU, see below) is enough to treat a skin area of two adult hand sizes.
                - Continue application until the skin is no longer red, itchy, or bumpy.
                - If the skin flares (i.e. becomes red, itchy, or bumpy), immediately start application again.
                - If you do not see any improvement in 2 weeks despite proper application, or experience pain,
                bleeding, fever or difficulty sleeping for a few nights, please contact a healthcare professional""")

            col1, col2 = st.columns([1,1])
            with col1:
                ftu_img = Image.open((os.path.join(image_dir, 'eczema', 'fingertip.jpg')))
                st.image(ftu_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>1 FTU can cover 2 adult hand areas</p>", unsafe_allow_html=True)

            st.subheader('Example brands')
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                desonide_img = Image.open((os.path.join(image_dir, 'eczema', 'desonide.jpg')))
                st.image(desonide_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>Desonide Cream</p>", unsafe_allow_html=True)
            with col2:
                betamethasone_img = Image.open((os.path.join(image_dir, 'eczema', 'betamethasone.png')))
                st.image(betamethasone_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>Betamethasone Valerate Cream</p>", unsafe_allow_html=True)
            with col3:
                elomet_img = Image.open((os.path.join(image_dir, 'eczema', 'elomet.jpg')))
                st.image(elomet_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>Mometasone Furoate Cream</p>", unsafe_allow_html=True)

        st.header("TLDR: Which medicines should I choose?")
        st.markdown("""
            - The gold standard for treating eczema is a combination of topical steroids and moisturizers.
            - Betamethasone 0.025% cream is a safe and effective option, and can be applied anywhere on the body (other than the face).
            - Hydrocortisone 1% cream is the most readily available but may not be strong enough for some people.
            - If you are experiencing itch, you can also add on oral antihistamines, taken when necessary.""")

# Fungal Skin Infection
if sidebar == 'Fungal Skin Infection':
    # Fungal Skin Infection Condition Tab
    if menu_bar == 'Condition':
        st.header('What is fungal skin infection?')
        st.write("""A fungal infection (mycosis) is a skin disease caused by a fungus. A fungus is a tiny organism, such as mold or 
            mildew. Fungi live everywhere. They can be found in soil, in the air, or even inside the body. These microscopic 
            organisms typically don't cause any problems for our skin.""")
        st.write("""Since fungi thrive in warm, moist environments, fungal skin infections can often develop in sweaty or damp areas 
            that don't get much airflow. Some examples include the feet, groin, and skin folds.""")

        st.header('What causes a fungal rash?')
        st.write("""When your skin comes into contact with a harmful fungus, the infection can cause the rash to appear. 
            The fungi can be spread to human in four ways:""")
        st.markdown("""
        - Human to human
        - Animal to human
        - Object to human (e.g. clothes, gyms, hot tubs)
        - Soil to human""")

        st.header('Types of fungal rashes')
        st.write('The fungal rash is named differently based on which body part is affected.')
        st.markdown("""
            - Tinea pedis (Athlete's foot): Fungal infection of your foot. It happens often to people who wear tight shoes, don't change their sweaty socks, and who use public baths and pools.
            - Tinea cruris (Jock itch): Rash on the groin.
            - Tinea capitis (Scalp ringworm): Rash on the scalp, mainly occuring in children.
            - Tinea corporis (Ringworm): Rash on the body.
            - Tinea manuum (Ringworm): Rash on the hands.""")

        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            tinea_corporis_img = preprocess(os.path.join(image_dir, 'fungal', 'corporis.jpg'), (500,500))
            st.image(tinea_corporis_img, use_column_width=True)
            st.caption("<p style='text-align:center; font-size:16px;'>Tinea Corporis</p>", unsafe_allow_html=True)
        with col2:
            tinea_manuum_img = preprocess(os.path.join(image_dir, 'fungal', 'manuum.jpeg'), (500,500))
            st.image(tinea_manuum_img, use_column_width=True)
            st.caption("<p style='text-align:center; font-size:16px;'>Tinea Manuum</p>", unsafe_allow_html=True)
        with col3:
            tinea_cruris_img = preprocess(os.path.join(image_dir, 'fungal', 'cruris.jpg'), (500,500))
            st.image(tinea_cruris_img, use_column_width=True)
            st.caption("<p style='text-align:center; font-size:16px;'>Tinea Cruris</p>", unsafe_allow_html=True)

        st.header('Symptoms')
        st.write("""These infections usually appear as a scaly, discolored and itchy skin rash. These patches typically appear red on 
            lighter skin or brown-gray on darker skin.""")
        st.write('Some symptoms include:')
        st.markdown("""
            - Scaly skin
            - Redness
            - Itching
            - Swelling
            - Blisters
            - Patches that resemble a ring with deeper colour on the outside
            - Patches with edges that are defined and raised
            - Overlapping rings""")

        st.header('What is the outlook for fungal skin infections?')
        st.markdown(
            """
            - Most fungal skin infections can be treated with over-the-counter or prescription creams.
            - Severe infections may require oral medications.
            """)


    # Fungal Skin Infection Management Tab
    elif menu_bar == 'Management':
        st.header('Pharmacological Therapies')
        st.subheader('Topical Antifungals')
        st.markdown("""
            - Localised fungal infections may respond to topical antifungal agents such as clotrimazole or miconazole.
            - Application needs to include an adequate margin around the lesion and be continued at least 1-2 weeks after the visible rash has cleared.
            - Recurrence is common.
            - Many different formulations: creams, powders, ointments, gels""")

        st.subheader('Oral Antifungals')
        st.markdown("""
            - Alternative for patients with extensive skin involvement and patients who fail topical therapy.
            - Some common examples include terbinafine and itraconazole.""")

        st.markdown("---")

        st.header('Tips for Managing Fungal Rashes')
        st.markdown("""
            - Do not share unwashed clothes, sports gear or towels with others.
            - Change your socks and undergarments at least once daily.
            - Keep your skin dry and clean.
            - Always wear slippers or sandals when at public spaces.
            """)        
  

    # Fungal Skin Infection Medications Tab
    elif menu_bar == 'Medications':
        st.header("Over-the-counter Medicines")
        st.write("""These medicines can be bought off the shelf at retail pharmacies such as Guardian,
            Watsons, Unity, and certain hospitals.""")
        # Topical Antifungals
        with st.expander(label="Topical Antifungals", expanded=False):
            st.subheader("What are topical antifungals?")
            st.markdown("""
                - Topical antifungals help to eradicate the fungi that are causing your skin infection.
                - **Examples of topical antifungals include clotrimazole, ketoconazole, isoconazole and miconazole**.
                """)

            st.subheader("Counselling Points")
            st.markdown("""
                - Apply a thin layer to the affected area, rub in gently, two times a day.
                - Ensure to include an adequate margin around the border of the affected region.
                - Continue for about one-two weeks after the disappearance of the symptoms.""")

            st.subheader('Example brands')
            col1, col2 = st.columns([1,1])
            with col1:
                clotrimazole_img = preprocess(os.path.join(image_dir, 'fungal', 'clotrimazole.jpg'), (5000,5000))
                #clotrimazole_img = Image.open((os.path.join(image_dir, 'fungal', 'clotrimazole.jpg')))
                st.image(clotrimazole_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>Clotrimazole Cream</p>", unsafe_allow_html=True)
            with col2:
                miconazole_img = preprocess(os.path.join(image_dir, 'fungal', 'miconazole.jpg'), (5000,5000))
                # miconazole_img = Image.open((os.path.join(image_dir, 'fungal', 'miconazole.jpg')))
                st.image(miconazole_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>Miconazole Cream</p>", unsafe_allow_html=True)

        # Oral Antihistamines
        with st.expander(label="Oral Antihistamines", expanded=False):
            st.subheader("What are oral antihistamines?")
            st.markdown("""
                - Oral antihistamines block the release of histamines, which is a potent inflammatory mediator commonly associated with allergic reactions.
                - They may be useful for temporary relief of itch in eczema.
                - **Examples include loratadine and cetirizine.**""")

            st.subheader("Counselling Points")
            st.markdown("""
                - (For Zyrtec-R/Clarityn) Take one tablet once a day for the relief of itch.""")

            st.subheader('Example brands')
            col1, col2 = st.columns([1,1])
            with col1:
                zyrtec_img = Image.open((os.path.join(image_dir, 'eczema', 'zyrtec.jpg')))
                st.image(zyrtec_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>Zyrtec-R (Cetirizine)</p>", unsafe_allow_html=True)
            with col2:
                clarityn_img = Image.open((os.path.join(image_dir, 'eczema', 'clarityn.jpg')))
                st.image(clarityn_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>Clarityn (Loratadine)</p>", unsafe_allow_html=True)

        st.header("TLDR: Which medicines should I choose?")
        st.markdown("""
            - It is essential to apply the antifungal cream/ointment/gel.
            - If you are experiencing itch, you may add on oral antihistamines or topical steroids, but stop when no longer needed.""")

# Psoriasis
if sidebar == 'Psoriasis':
    # Psoriasis Condition Tab
    if menu_bar == 'Condition':
        st.header('What is psoriasis?')
        st.write("""Psoriasis is a skin disorder, where skin cells multiply faster than normal. This makes the skin build up into 
            raised, itchy, and scaly patches, most commonly on the knees, elbows, trunk and scalp.""")
        st.write("""Psoriasis is a long-lasting disease. The condition tends to go through cycles, flaring for a few weeks/months 
            then subsiding for awhile.""")

        st.header('What causes psoriasis?')
        st.markdown("""The exact cause of psoriasis is still unknown. It is thought to be an immune system disorder that triggers new skin cells to form too quickly.
        Psoriasis is the result of a sped-up skin production process. Typically, skin cells grow deep in your skin and slowly rise to the surface. Eventually, they fall off.
        The typical life cycle of a skin cell is 1 month. In people with psoriasis, this production process may occur in just a few days.
        Because of this, skin cells don’t have time to fall off. This rapid overproduction leads to the buildup of skin cells.""")

        st.header('Types of Psoriasis')
        st.markdown("""
        #### Plaque psoriasis
        - Plaque psoriasis is the most common type of psoriasis (makes up 80-90\% of psoriasis).
        - It is characterised by dry, itchy, raised skin patches (plaques) covered with scales, and usually appears on the elbows, knees, lower back and scalp.

        #### Nail psoriasis
        - Nail psoriasis affects the fingernails and toenails, and may cause pitting, abnormal nail growth or discoloration.
        - Psoriatic nails may loosen, crumble and/or separate from the nail bed.

        #### Guttate psoriasis
        - Guttate psoriasis is marked by small, drop-shaped, scaling spots that are pink, red, brown or purple in colour, appearing on the body, arms or legs.
        - Primarily affects young adults and children.
        - Usually triggered by a bacterial infection like strep throat, tonsilitis and respiratory infections.

        #### Inverse psoriasis
        - Inverse psoriasis causes discolored, shiny lesions that appear on skin folds, such as the groin, buttocks, armpits and under the breasts.
        - Worsens with friction and sweating.
        - Fungal infections may trigger this type of psoriasis.

        #### Pustular psoriasis
        - Pustular psoriasis causes discolored, scaly skin with tiny pus-filled blisters.
        - It can occur in widepsread patches or on small areas on the palms or soles.

        #### Erythrodermic psorasis
        - Erythrodermic psoriasis can cover the entire body with a peeling rash that can itch or burn intensely.
        - It is the least common type of psoriasis.
        - It can be acute or chronic.
        - It needs to be treated immediately as it may be life threatening
        """)

        col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,1])
        col_list = [col1, col2, col3, col4, col5, col6]
        psoriasis_image_folder = os.path.join(image_dir, 'psoriasis')
        psoriasis_images = {os.path.join(psoriasis_image_folder, 'plaque.jpg'): 'Plaque Psoriasis',
                       os.path.join(psoriasis_image_folder, 'nail.jpg'): 'Nail Psoriasis',
                       os.path.join(psoriasis_image_folder, 'guttate.jpg'): 'Guttate Psoriasis',
                       os.path.join(psoriasis_image_folder, 'inverse.jpg'): 'Inverse Psoriasis',
                       os.path.join(psoriasis_image_folder, 'pustular.jpg'): 'Pustular Psoriasis',
                       os.path.join(psoriasis_image_folder, 'erythrodermic.jpg'): 'Erythrodermic Psoriasis'}

        for index, (image_path, caption) in enumerate(psoriasis_images.items()):
            with col_list[index]:
                psoriasis_image = preprocess(image_path, (1000,1000))
                st.image(psoriasis_image, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>{caption}</p>", unsafe_allow_html=True)

        st.header('Signs and Symptoms')
        st.write("""Psoriasis symptoms differ from person to person and depend on the type of psoriasis you have. Areas affected may be as small as a fwe flakes on your elbow,
            or cover the majority of your body.""")
        st.write('The most common symptoms of plaque psoriasis include:')
        st.markdown("""
            - Raised patches of skin that appears red on lighter skin or purplish on darker skin.
            - Flaky scales that may appear silver on lighter skin or gray on darker skin.
            - Dry, cracked skin that may bleed
            - Soreness, itching, and burning sensation around the patches
            - Thick, pitted nails
            - Painful and swollen joints""")
        st.write("Not every person will experience all these symptoms. Some people may present with 1 symptom whereas some may present with many")
        st.write("""Most people with psoriasis go through 'cycles' of symptoms. The condition may cause severe symptoms for a few days or weeks,
            then it gets better and may even completely disappear. The condition then flares up again after a certain period of time and the cycle repeats.""")

        st.header('What is the outlook for psoriasis?')
        st.markdown("""
            - Psoriasis can be a lifelong condition that can usually be controlled with treatment.
            - It may go away for a long time and then return.
            - With proper treatment, it will not affect your overall health.
            - There is a strong association between psoriasis and other health problems such as arthritis and heart disease.""")


    # Psoriasis Management Tab
    elif menu_bar == 'Management':
        st.header('Pharmacological Therapies')
        st.subheader('Emollients and Moisturizers')
        st.markdown("""
            - The regular use of moisturizers softens psoriasis and adds moisture to the skin. This improves dryness, scaling and irritation.
            - Thick ointments like white soft paraffin are often recommended for chronic plaques and hand/foot psoriasis.
            - They should be applied liberally and frequently.""")

        st.subheader('Topical Steroids')
        st.markdown("""
            - Topical steroids are safe and relatively easy to use for most types of psoriasis.
            - They also can be used in combination with other agents.
            - Potent steroids are more effective than mild topical steroids, but they have a higher risk of side effects.
            - Side effects include skin atrophy, striae (stretch marks) and telangiestasia.
            - They should be used with caution in large areas and for limited periods.""")

        st.subheader('Topical Vitamin D Analogues')
        st.markdown("""
            - Topical vitamin D analogues can help to reduce the thickness and scaliness of plaques.
            - A commonly used vitamin D analogue in Singapore is Calcipotriol.
            - Calcipotriol is available in combination with a potent topical steroid as a gel/ointment. This is commonly used as the first line 
            treatment in plaque psoriasis.""")

        st.subheader('Coal Tar')
        st.markdown("""
            - Coal tar is particularly useful for scalp psoriasis and large thin plaque psoriasis.
            - Side effects include skin irritation.
            - They can be messy as it can stain the skin, hair, and clothing, and has an associated odour.""")

        st.subheader('Other Medications')
        st.write('These are often used when the condition is more severe or is resistant to treatment.')
        st.markdown("""
            - Methotrexate
            - Acitretin
            - Cyclosporine
            - Secukinumab
            - Many others""")
        st.markdown("---")
        

        st.header('Non-pharmacological Therapies')
        st.subheader('Phototherapy')
        st.markdown("""
            - Phototherapy can be very effective in the treatment of psoriasis.
            - Generally reserved for cases where topical therapy is ineffective or too much of the skin is involved to treat with topical agents effectively.
            - Early side effects include sunburn and photosensitivity rashes.
            - Late side effects include ageing of the skin and skin cancer.""")
        st.markdown("---")


        st.header('Tips for Managing Psoriasis')
        st.markdown("""
            - Take daily baths
            - Apply moisturizer daily
            - Expose your skin to small amounts of sunlight
            - Avoid scratching
            - Stay cool
            - Avoid certain psoriasis triggers like smoking, skin injuries, intense sun exposure""")      

    # Psoriasis Medications Tab
    elif menu_bar == 'Medications':
        st.header("Self-treatment is not recommended.")

# Warts
if sidebar == 'Warts':
    # Warts Condition Tab
    if menu_bar == 'Condition':
        st.header('What are Warts?')
        st.write("""Warts are a common viral infection of the skin that is caused by the human papillomavirus (HPV). """)

        st.header('What causes Warts?')
        st.markdown("""When HPV enters a cut in the skin, it causes a skin infection that forms warts. Warts are very contagious. The virus can spread from person to person
        by skin-to-skin contact or from the environment (e.g. swimming pool decks, changing room floors). They also can spread from one area of the body to another.""")

        st.header('Types of Warts')
        st.markdown("""
        #### Common warts
        - Common warts affect the hands. They are called common warts as they are the most common type.

        #### Flat warts
        - Flat warts occur commonly on the face, neck, arms and legs.
        - Usually smooth, flesh-coloured, pink or brown, flat-topped papules.

        #### Plantar warts
        - Plantar warts appears on the soles of the feet.
        - They resemble calluses but with tiny black dots in the center.
        - Often painful on pressure.

        #### Ano-genital warts
        - Ano-genital warts form on the penis, vagina, or rectum.
        - These warts are a type of sexually transmitted infection.

        #### Periungual and subungual warts
        - These warts form under or around fingernails and toenails.
        """)

        col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
        col_list = [col1, col2, col3, col4, col5]
        warts_image_folder = os.path.join(image_dir, 'warts')
        warts_images = {os.path.join(warts_image_folder, 'common.png'): 'Common Wart',
                       os.path.join(warts_image_folder, 'flat.png'): 'Flat Wart',
                       os.path.join(warts_image_folder, 'plantar.png'): 'Plantar Wart',
                       os.path.join(warts_image_folder, 'genital.png'): 'Ano-genital Wart',
                       os.path.join(warts_image_folder, 'periungual.png'): 'Periungual Wart'}

        for index, (image_path, caption) in enumerate(warts_images.items()):
            with col_list[index]:
                warts_image = preprocess(image_path, (1000,1000))
                st.image(warts_image, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>{caption}</p>", unsafe_allow_html=True)

        st.header('Signs and Symptoms')
        st.write("""Common warts usually occur on your fingers or hands, and may be:""")
        st.markdown("""
            - Small, fleshy, grainy bumps.
            - Rough.
            - Sprinkled with black pinpoints, which are small clotted blood vessels.""")

        st.header('What is the outlook for Warts?')
        st.markdown(
            """
            - Warts are generally not dangerous and typically respond to over-the-counter treatment.
            - Once you have the virus, there’s no sure way to keep warts from returning.
            - After treatment, warts can reappear at the same location or a different part of the body. But some people get rid of warts and never have one again.
            """)

    # Warts Management Tab
    elif menu_bar == 'Management':
        st.header('Pharmacological Therapies')
        st.subheader('Salicylic Acid')
        st.markdown("""
            - Topical salicylic acid exfoliates the affected epidermis and stimulates local immunity.
            - It is easy to apply, painless, and has low risk of serious side effects.
            - Local skin irritation is common.""")

        st.markdown("---")

        st.header('Non-pharmacological Therapies')
        st.subheader('Cryotherapy')
        st.markdown("""
            - Cryotherapy with liquid nitrogen is a common treatment for warts.
            - A disadvantage of cryotherapy is the pain associated with the treatment.
            - Treatment is repeated every 1-3 weeks until wart resolution.
            - Cryotherapy is often combined with salicylic acid treatment in an attempt to augment efficacy.""")
        st.markdown("---")

        st.header('Tips for Managing Warts')
        st.markdown("""
            - Do not pick or scratch at warts
            - Wear slippers or sandals in public showers, locker rooms, and pool areas
            - Do not touch someone’s wart
            - Keep warts dry as moisture tends to allow warts to spread""")

    # Warts Medications Tab
    elif menu_bar == 'Medications':
        st.markdown('**Please consult a healthcare professional if the face, nails, anus or genitalia is involved.**')
        st.header("Over-the-counter Medicines")
        st.write("""These medicines can be bought off the shelf at retail pharmacies such as Guardian,
            Watsons, Unity, and certain hospitals.""")
        with st.expander(label="Salicylic Acid", expanded=True):
            st.subheader("What is Salicylic Acid?")
            st.markdown("""
                - Salicylic acid works by softening the skin of the wart, causing it to peel off. It can also stimulate local immunity.
                """)

            st.subheader("Counselling Points")
            st.markdown("""
                - Soak the wart in hot water for 5 minutes, then dry the area.
                - Rub the top of the warts with a pumice stone or a nail file.
                - Apply the medicine directly to the wart, once or twice daily (depending on the product) before bedtime. Try to avoid application on normal skin.
                - Let the solution dry fully.
                - Cover the wart with a plaster if possible.
                - It may take 4-8 weeks for visible improvement.
                - Local skin irritation is common and expected. If significant, reduce the frequency of application.""")

            st.subheader('Example brands')
            col1, col2 = st.columns([1,1])
            with col1:
                #duofilm_img = preprocess(os.path.join(image_dir, 'warts', 'clotrimazole.jpg'), (5000,5000))
                duofilm_img = Image.open((os.path.join(image_dir, 'warts', 'duofilm.jpg')))
                st.image(duofilm_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>Duofilm Solution</p>", unsafe_allow_html=True)
            with col2:
                #miconazole_img = preprocess(os.path.join(image_dir, 'fungal', 'miconazole.jpg'), (5000,5000))
                collomack_img = Image.open((os.path.join(image_dir, 'warts', 'collomack.jpg')))
                st.image(collomack_img, use_column_width=True)
                st.caption(f"<p style='text-align:center; font-size:16px;'>Collomack Solution</p>", unsafe_allow_html=True)







# Citations
# https://www.elevatedaestheticsspa.com/blog/2018/3/7/what-type-of-acne-do-you-have (acne images)
# my.clevelandclinic..org
# mayoclinic
# webmd
# healthline