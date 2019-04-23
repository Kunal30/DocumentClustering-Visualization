#!/bin/bash

if [ "$1" == "create-lda" ]; then
	echo 'BUILDING UP THE LDA Model....'
	python getLDAModel.py

elif [ "$1" == "apply-tsne" ]; then
	echo 'Applying TSNE on the document & topic matrix.......'
	echo 'Creating latent semantics vector files............'
	echo 'Generating the Z coordinates..............'
	echo 'Gathering the best topic for every document........'
	python lda2tsne.py

elif [ "$1" == "apply-pca" ]; then
	echo 'Applying TSNE on the document & topic matrix.......'
	echo 'Creating latent semantics vector files............'
	echo 'Generating the Z coordinates..............'
	echo 'Gathering the best topic for every document........'
	python lda2pca.py

elif [ "$1" == "visualize-3d" ]; then
	echo 'Setting up environment for 3D visualization........'
	python 3D_Visualization.py

elif [ "$1" == "run-project-tsne" ]; then
	echo 'Running the whole project from the beginning.......'
	echo 'BUILDING UP THE LDA Model....'
	python getLDAModel.py
	echo 'Applying TSNE on the document & topic matrix.......'
	echo 'Creating latent semantics vector files............'
	echo 'Generating the Z coordinates..............'
	echo 'Gathering the best topic for every document........'
	python lda2tsne.py
	echo 'Setting up environment for 3D visualization........'
	python 3D_Visualization.py

elif [ "$1" == "run-project-pca" ]; then
	echo 'Running the whole project from the beginning.......'
	echo 'BUILDING UP THE LDA Model....'
	python getLDAModel.py
	echo 'Applying TSNE on the document & topic matrix.......'
	echo 'Creating latent semantics vector files............'
	echo 'Generating the Z coordinates..............'
	echo 'Gathering the best topic for every document........'
	python lda2pca.py
	echo 'Setting up environment for 3D visualization........'
	python 3D_Visualization.py


else
	echo "use \"create-lda\" argument to create the lda model"
	echo "use \"apply-tsne\" argument to  apply tsne and generate the dependencies"
	echo "use \"apply-pca\" argument to  apply tsne and generate the dependencies"
	echo "use \"visualize-3d\" argument to generate 3D visualization"
	echo "use \"run-project-tsne\" argument to run the above 3 steps/commands with tsne sequentially from scratch"
	echo "use \"run-project-pca\" argument to run the above 3 steps/commands with pca sequentially from scratch"
fi	
	