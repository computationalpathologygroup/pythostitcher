<h1 align="center">PythoStitcher</h2>
<p align="center">
   <a href="https://github.com/psf/black"><img alt="empty" src=https://img.shields.io/badge/code%20style-black-000000.svg></a>
   <a href="https://github.com/computationalpathologygroup/pythostitcher/releases"><img alt="empty" src=https://img.shields.io/github/v/release/computationalpathologygroup/pythostitcher?include_prereleases&label=pre-release&logo=github.svg></a>
</p>

    
## What is PythoStitcher?
Pythostitcher is a tool inspired by [AutoStitcher](https://www.nature.com/articles/srep29906) to stitch histopathology images into an artifical whole mount image. These artificial whole-mounts are indispensable for multimodal imaging research, since they greatly improve the ability to relate histopathology information to pre-operative imaging. PythoStitcher works fully automatically and is able to generate very high resolution (0.25 µm/pixel) whole-mounts. 

<p align="center">
  <img width="640" height="480" src="./img/tform_progression.gif">
</p>

## Does PythoStitcher also work on my data?
If your data consists of either halves or quadrants of a certain type of tissue (i.e. a prostate), PythoStitcher should be able to reconstruct the artificial whole-mount for you. PythoStitcher expects that your data has multiple resolution layers, also known as a pyramidal file, and preferably uses .mrxs or .tiff files. In addition, PythoStitcher requires a tissue mask of said tissue. This tissue mask can be generated by your tissue segmentation algorithm of choice; in the provided sample data we make use of the algorithm from [Bándi et al](https://pubmed.ncbi.nlm.nih.gov/31871843/).

## How do I run PythoStitcher?
#### Docker container 
It is highly recommended to run PythoStitcher as a Docker container, since PythoStitcher uses some libraries that need to be built from source. The Docker container comes prepackaged with these libraries, as well as with model weights of the involved CNNs, and should run out-of-the-box. You can pull the container with the following command or alternatively build it yourself locally with the provided Dockerfile in /build.

	docker pull ghcr.io/computationalpathologygroup/pythostitcher:latest

#### Data preparation
Your input data should be prepared as follows, where you make a separate raw_images and raw_masks directory for your high resolution image and tissue mask files, respectively. Ensure that the name of the tissue mask is exactly the same as that of the image. If you want to enforce the location of each fragment in the final reconstruction, you can include a force_config.txt file. See the example_force_config.txt file on how to format this. If you leave out this file, PythoStitcher will automatically determine the optimal configuration of the tissue fragments.
	
	data/ 
	└── patient_ID
	|    └── raw_images
	|        ├── image1
	|        └── image2
	|    └── raw_masks
	|        ├── image1
	|        └── image2
	│    └── force_config.txt [OPTIONAL]


#### Usage instructions
            
After preparing the input data in the aforementioned format, you can run PythoStitcher through the command line using:

    docker run pythostitcher --datadir "/path/to/data" --savedir "/path/to/save/results" --resolution x
where *datadir* refers to the directory with your input data, *savedir* refers to the location to save the result and *resolution* refers to the resolution in µm/pixel at which you want to save the final reconstruction. Example line to obtain the result for patient P000001 from our sample prostate dataset:

    docker run pythostitcher --datadir "~/sample_data/prostate_P000001" --savedir "~/results" --resolution 0.25

#### Sample data 
If you don't have any data available, but are still curious to try PythoStitcher, you can make use of our sample data available from <a href="https://zenodo.org/record/7636102"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7636102.svg" alt="DOI"></a>. The sample data includes two prostatectomy cases, one case with four fragments and one case with two fragments. 

## Acknowledgements
The development of PythoStitcher would not have been possible without the open-sourcing of [JigsawNet](https://github.com/Lecanyu/JigsawNet), [ASAP](https://github.com/computationalpathologygroup/ASAP) and [PyVips](https://github.com/libvips/pyvips).

## Licensing
The source code of Pythostitcher is licensed under the [GNU Lesser General Public License (LGPL)](https://www.gnu.org/licenses/lgpl-3.0.nl.html). The provided sample data is licensed under the [CC Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/legalcode). Please take these licenses into account when using PythoStitcher.

