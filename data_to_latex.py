import pandas as pd
import numpy as np
import os

df = pd.read_csv('data.csv')
df = df.round(2)

image_texts = np.array([])

temp_directory = os.path.join(os.path.dirname(__file__), 'temp')
if not os.path.exists(temp_directory):
    os.makedirs(temp_directory)

for image_name in list(set(df['image_name'])):
    image_df = df[df['image_name'] == image_name]
    image_df['denoise_lpg_pca'] = image_df['denoise_psnr_lpg_pca'].map(str) + '(' +  image_df['denoise_ssim_lpg_pca'].map(str) + ')'
    image_df['denoise_mf'] = image_df['denoise_psnr_mf'].map(str) + '(' +  image_df['denoise_ssim_mf'].map(str) + ')'
    image_df['denoise_nlm'] = image_df['denoise_psnr_nlm'].map(str) + '(' +  image_df['denoise_ssim_nlm'].map(str) + ')'
    image_df['denoise_bm3d'] = image_df['denoise_psnr_bm3d'].map(str) + '(' +  image_df['denoise_ssim_bm3d'].map(str) + ')'
    image_df = image_df[['sigma', 'denoise_lpg_pca', 'denoise_mf', 'denoise_nlm', 'denoise_bm3d']]
    image_df['sigma'] = image_df['sigma'].map(int)
    image_df.columns = ['sigma', 'LPG-PCA', 'MF', "NLM", 'BM3D']

    path = os.path.join(temp_directory, image_name + '.tex')
    image_df.to_latex(path, index=False, column_format='lrrrr')

    with open(path, 'r') as file:
        image_text = file.read()
        image_text = image_text.replace(' ', '').replace(r'\toprule',r'\toprule &&' + image_name + r'\\ \midrule')
        image_text = r'\noindent\begin{minipage}{.5\linewidth}' + '\n' + image_text + '\n' + r'\end{minipage}'
        image_text = image_text.replace('\n\n', '\n').replace('sigma&', '\\sigma&')
        image_texts = np.append(image_texts, image_text)
    
    os.remove(path)

result = '\n'.join(image_texts)

filename = 'tables.tex'
with open(filename, "w+") as file:
    file.write(result)

if(len(os.listdir(temp_directory))) == 0:
    os.rmdir(temp_directory)
