function faceRecognitionGUI()
    % Membuat GUI Figure
    fig = uifigure('Name', 'Face Recognition with Fisherfaces', 'Position', [100 100 600 400]);

    % Button untuk Memilih Gambar Uji
    btnSelectImage = uibutton(fig, 'push', 'Text', 'Select Test Image', 'Position', [50 300 200 50]);
    btnSelectImage.ButtonPushedFcn = @(btn, event) selectTestImage();

    % Axes untuk Menampilkan Preview Gambar Uji
    axPreview = uiaxes(fig, 'Position', [300 150 250 200]);
    title(axPreview, 'Test Image Preview');

    % Label untuk Menampilkan Hasil Pengakuan
    lblResult = uilabel(fig, 'Text', 'Recognized Person: ', 'Position', [50 250 400 30], 'FontSize', 14);

    % Label untuk Menampilkan Tingkat Akurasi
    lblAccuracy = uilabel(fig, 'Text', 'Accuracy: ', 'Position', [50 200 400 30], 'FontSize', 14);

    % Load Database
    [meanFace, V_pca, V_lda, X_lda, labels] = loadDatabase();

    % Callback untuk Button Memilih Gambar Uji
    function selectTestImage()
        % Pilih Gambar Uji
        [file, path] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files (*.jpg, *.png, *.bmp)'}, 'Select Test Image');
        
        if isequal(file, 0)
            return;  % Batal jika tidak ada file yang dipilih
        end
        
        % Load Gambar dan Tampilkan di Preview
        testImage = imread(fullfile(path, file));
        testImageGray = rgb2gray(testImage);  % Convert to grayscale
        testImageFiltered = imgaussfilt(testImageGray, 2);  % Apply Gaussian filter
        testImageResized = imresize(testImageFiltered, [100, 100]);  % Resize to 100x100
        
        % Tampilkan gambar yang sudah diproses
        imshow(testImageResized, 'Parent', axPreview);

        % Proses Pengakuan Wajah
        [recognizedLabel, accuracy] = recognizeFace(testImageResized, meanFace, V_pca, V_lda, X_lda, labels);

        % Tampilkan Hasil di GUI
        lblResult.Text = ['Recognized Person: ', recognizedLabel];
        lblAccuracy.Text = ['Accuracy: ', num2str(accuracy), '%'];
    end
end

function [meanFace, V_pca, V_lda, X_lda, labels] = loadDatabase()
    % Load images and labels
    dataFolder = 'Dataset';
    imageFiles = dir(fullfile(dataFolder, '**/*.jpg'));
    numFiles = length(imageFiles);
    images = {};
    labels = {};
    targetSize = [100, 100];  % Ensure all images are 100x100

    % Load images and labels
    for i = 1:numFiles
        img = imread(fullfile(imageFiles(i).folder, imageFiles(i).name));
        imgGray = rgb2gray(img);  % Convert to grayscale
        imgFiltered = imgaussfilt(imgGray, 2);  % Noise reduction
        imgResized = imresize(imgFiltered, targetSize);  % Resize to 100x100
        images{i} = imgResized;
        labels{i} = imageFiles(i).folder;  % Assuming folder name as label
    end

    % Prepare image matrix X
    [numRows, numCols] = size(images{1});
    numPixels = numRows * numCols;
    X = zeros(numPixels, numFiles);

    for i = 1:numFiles
        X(:, i) = reshape(images{i}, numPixels, 1);  % Reshape each image to a column vector
    end

    % Compute mean face and center data
    meanFace = mean(X, 2);
    X_centered = X - meanFace;

    % PCA
    [U, S, ~] = svd(X_centered, 'econ');
    singularValues = diag(S);
    energy = cumsum(singularValues.^2) / sum(singularValues.^2);
    k = find(energy >= 0.95, 1);  % Select components to reach 95% variance

    % Ensure minimum components if threshold not met
    if isempty(k)
        k = min(10, size(U, 2));
    end

    V_pca = U(:, 1:k);  % Select top k eigenfaces
    X_pca = V_pca' * X_centered;

    % LDA: Calculate between-class (Sb) and within-class (Sw) scatter matrices
    uniqueLabels = unique(labels);
    numClasses = length(uniqueLabels);
    Sb = zeros(size(X_pca, 1));
    Sw = zeros(size(X_pca, 1));

    for i = 1:numClasses
        classIndices = find(strcmp(labels, uniqueLabels{i}));
        classMean = mean(X_pca(:, classIndices), 2);
        Sb = Sb + length(classIndices) * (classMean - mean(X_pca, 2)) * (classMean - mean(X_pca, 2))';

        for j = classIndices
            Sw = Sw + (X_pca(:, j) - classMean) * (X_pca(:, j) - classMean)';
        end
    end

    % Add regularization to Sw to avoid singularity
    epsilon = 1e-5;
    Sw = Sw + epsilon * eye(size(Sw));

    % Solve for LDA projection
    [V_lda_full, D] = eig(Sb, Sw);
    [~, idx] = sort(diag(D), 'descend');
    V_lda_full = V_lda_full(:, idx);

    % Select components for LDA
    numLDAComponents = min(numClasses - 1, size(V_lda_full, 2));
    V_lda = V_lda_full(:, 1:numLDAComponents);
    X_lda = V_lda' * X_pca;  % Project to LDA space

    % Check dimensions for debugging
    disp(['Dimensi V_pca: ', num2str(size(V_pca))]);
    disp(['Dimensi V_lda: ', num2str(size(V_lda))]);
end

function [recognizedLabel, accuracy] = recognizeFace(testImage, meanFace, V_pca, V_lda, X_lda, labels)
    % Flatten the test image and center it
    testVector = double(reshape(testImage, [], 1));
    
    % Ensure meanFace is a column vector of the same size
    meanFace = reshape(meanFace, [], 1);
    fprintf('Dimensi testVector: %d %d\n', size(testVector));
    fprintf('Dimensi meanFace: %d %d\n', size(meanFace));

    % Check if dimensions are compatible
    if size(testVector, 1) ~= size(meanFace, 1)
        error('Dimensi meanFace tidak cocok dengan testVector. Pastikan keduanya memiliki jumlah piksel yang sama.');
    end

    % Center the test vector
    testVector_centered = testVector - meanFace;

    % Apply PCA transformation
    testVector_pca = V_pca' * testVector_centered;

    % If LDA is available, apply it
    if ~isempty(V_lda)
        % If LDA components are available, apply LDA transformation
        testVector_lda = V_lda' * testVector_pca;
        distances = vecnorm(X_lda - testVector_lda, 2, 1); % Use LDA if available
    else
        % Use PCA only if LDA is not available
        distances = vecnorm(X_lda - testVector_pca, 2, 1); % Use PCA space only
    end

    % Find the closest match
    [~, minIndex] = min(distances);
    recognizedLabel = labels{minIndex};
    accuracy = 1 / (1 + distances(minIndex));  % Optional accuracy measure
end



