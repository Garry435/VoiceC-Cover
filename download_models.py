from pathlib import Path
import requests

MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
VoiceC_DOWNLOAD_LINK = 'https://huggingface.co/Garry908/sample-test/resolve/main/'

mdxnet_models_dir = 'mdxnet_models'

def dl_model(link, model_name):
    with requests.get(f'{link}{model_name}') as r:
        r.raise_for_status()
        with open(f"{model_name}", 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == '__main__':
    mdx_model_names = ['UVR-MDX-NET-Voc_FT.onnx', 'UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx']
    for model in mdx_model_names:
        print(f'Downloading {model}...')
        dl_model(MDX_DOWNLOAD_LINK, f"{mdxnet_models_dir}/{model}")

    VoiceC_model_names = ['hubert_base.pt', 'rmvpe.pt']
    for model in VoiceC_model_names:
        print(f'Downloading {model}...')
        dl_model(VoiceC_DOWNLOAD_LINK, model)

    print('All models downloaded!')
