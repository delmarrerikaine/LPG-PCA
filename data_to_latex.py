import pandas as pd

df = pd.read_csv('data.csv')

df_psnr = df[['image_name', 'sigma', 'denoise_psnr_lpg_pca', 'denoise_psnr_mf', 'denoise_psnr_nlm', 'denoise_psnr_bm3d']]
df_psnr.columns = ['image', 'sigma', 'LPG-PCA', 'MF', 'NLM', 'BM3D']
df_psnr = df_psnr.round(3)
df_psnr.to_latex('psnrs.tex', index=False)

df_ssim = df[['image_name', 'sigma', 'denoise_ssim_lpg_pca', 'denoise_ssim_mf', 'denoise_ssim_nlm', 'denoise_ssim_bm3d']]
df_ssim.columns = ['image', 'sigma', 'LPG-PCA', 'MF', 'NLM', 'BM3D']
df_ssim = df_ssim.round(3)
df_ssim.to_latex('ssims.tex', index=False)