%%%%%%%%%%%%%%%%%%%%%%% AMATH582 HW4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: Han Song
% Class: AMATH 582
% Due Date: 3/6/2020
clear all;clc; close all
%% Part 1: Yale Faces B

% load files to establish number of iterations. Please extract cropped
% images folder into working folder with name = "CroppedYale" and uncropped
% images folder into working folder with name = "yalefaces" for code to
% work smoothly
Cropped_folder = dir('CroppedYale');
Cropped_folder = Cropped_folder(3:end);
Uncropped_folder = dir('yalefaces');
Uncropped_folder = Uncropped_folder(3:end);
%
% Consolidate all images into 1 matrix, each column is 1 image
index = 1; %dummy index
image_size = 192*168;
all_img = zeros(image_size,2432); %2432 is total number of images in all folders
ave_face = zeros(image_size,length(Cropped_folder(3:end)));
for i=1:length(Cropped_folder)
    subfolder = dir(['CroppedYale/' Cropped_folder(i).name]);
    for j=3:length(subfolder)
        data = imread(['CroppedYale/' Cropped_folder(i).name '/' subfolder(j).name]);
        data = reshape(data,image_size,1);
        all_img(:,index) = data;
        index = index+1;
    end
    ave_face(:,i) = sum(all_img,2)/length(all_img);
end

% SVD and plot spectrum
all_img = double(all_img);
[U,S,V] = svd(all_img,'econ');
figure(1)
plot(diag(S)/sum(diag(S)),'bo')
ylabel('Singular Values')
xlabel('Average Faces Images')
title('Singular Value Spectrum on Cropped Images')
%
% Testing 1 random image and multiple images
% 1 image
random_img = imread('CroppedYale/yaleB18/yaleB18_P00A-020E+10.pgm');
figure(3)
subplot(2,3,1)
imshow(random_img)
title('Test Image')
random_img = double(reshape(random_img,192*168,1));
rank = [4,10,50,150,250];
for i = 1:length(rank)
    U_rec = U(:,1:rank(i));
    recon = U_rec*U_rec'*random_img;
    recon = reshape(recon,192,168);
    subplot(2,3,i+1)
    imshow(uint8(recon))
    title(['rank ',num2str(rank(i))])
end
%
% multiple images
img_no = [5,25,125,625]; % random images order
rank = [5, 50, 200];
figure(2)
for i = 1: length(img_no)
    for j = 1:length(rank)
        reconstruct = U*S(:,1:rank(j))*V(:,1:rank(j))';
        subplot(length(rank)+1,length(img_no),(j-1)*length(img_no)+i)
        imshow(uint8(reshape(reconstruct(:,img_no(i)),192,168)));
        if j == 1 
            title(['Image ' num2str(img_no(i))])
        end
        if i == 1 
            ylabel(['Reconstruct rank ' num2str(rank(j))])
        end
    end
    subplot(length(rank)+ 1,length(img_no),j*length(img_no)+ i)
    imshow(uint8(reshape(all_img(:,img_no(i)),192,168)));
    if i == 1
        ylabel('Original')
    end
end
            
%
% Original
image_size_uncropped = 243*320;
original = zeros(image_size_uncropped,165);
ave_face_original = original;
for i=1:length(Uncropped_folder)
    data = imread(['yalefaces/' Uncropped_folder(i).name]);
    data = reshape(data,image_size_uncropped,1);
    original(:,i) = data;
    ave_face_original(:,i) = sum(original,2)/length(original);
end

% SVD and plot
original = double(original);
[U_O,S_O,V_O] = svd(original,'econ');
figure(4)
plot(diag(S_O)/sum(diag(S_O)),'o')
ylabel('Singular Values')
xlabel('Images')

% Test multiple image with multiple rank
reconstruct = U_O*S_O(:,1:50)*V_O(:,1:50)';
img_no = [5,10,50,100,150];
for i = 1:length(img_no)
    subplot(2,length(img_no),i)
    imshow(uint8(reshape(original(:,img_no(i)),243,320)));
    title('Original Image')
    subplot(2,length(img_no),length(img_no)+i)
    imshow(uint8(reshape(reconstruct(:,img_no(i)),243,320)));
    title('Reconstructed...')
end

% similar to the cropped images, size image exceeds matlab limit and cant
% test.

% rand_img = imread('yalefaces/subject08.wink');
% figure(6)
% subplot(2,3,1)
% imshow(rand_img)
% title('Test Image')
% rand_img = double(reshape(rand_img,243*320,1));
% rank = [2];
% for i = 1:length(rank)
%     U_rec = U_O(:,1:rank(i));
%     recon = reshape(U_rec*U_rec'*rand_img,243,320);
%     subplot(2,3,i+1)
%     imshow(uint8(recon))
%     title(['rank ',num2str(rank(i))])
% end

%% Part 2: Music Classification

% getting data...
[Songs , info_str, song_list] = getTrainingSet('Test2');
info_int = str2double(info_str(:,2:end));
sample_size = info_int(1,2);
num_song = info_int(end,end);
% set label for each artist n = 1,2,3,...
song_matrix = ones(1,num_song); 
for i = 1:length(info_int(:,1))
    if i ~= 1
        song_matrix(:,info_int(i-1,3)+1:info_int(i,3)) = i;
    end
end

% Separate training and testing set
num_test = 5; % test 5 songs to get avg accuracy for KNN and SVM
random_index = randi([1 num_song],1,num_test);
test_song = Songs(:,random_index);
actual_label = song_matrix(:,random_index);
Songs(:,random_index) = [];
song_matrix(:,random_index) = [];
trainingSet = Songs;
target = song_matrix;

% Obtain spectrogram of training and testing set
spec_train = zeros(sample_size,num_song - num_test);
for i = 1:length(trainingSet(1,:))
    ft = fft(trainingSet(:,i));
    spec = abs(fftshift(ft));
    spec_train(:,i) = spec(:,1);
end

spec_test = zeros(sample_size, 1);
for i = 1:num_test
    ft = fft(test_song(:,i));
    spec = abs(fftshift(ft));
    spec_test(:,i) = spec(:,1);
end

[a,b]=size(spec_train); % compute data size
ab=mean(spec_train,2); % compute mean for each row
spec_train=spec_train-repmat(ab,1,b); % subtract mean

[c,d]=size(spec_test); % compute data size
cd=mean(spec_test,2); % compute mean for each row
spec_test=spec_test-repmat(cd,1,d); % subtract mean

% SVD
[~,S,V] = svd(spec_train','econ');
figure(5)
plot(diag(S)/sum(diag(S)),'bo')
xlabel('Songs')
ylabel('Singular values')
title('Singular Value Spectrum for Test 1')

%
% KNN
knn.mod = fitcknn(V',target,'NumNeighbors',5);
label = predict(knn.mod,test_song');
accuracy = 0;
for i = 1:length(label)
    if label(i) == actual_label(i)
        accuracy = accuracy + 1;
    end
end
accuracy = accuracy/num_test;
fprintf('Accuracy for test 1 using KNN is %.1f \n', accuracy);

%
% SVM
svm.mod = fitcecoc(V',target);
label_svm = predict(svm.mod,test_song');
accuracy = 0;
for i = 1:length(label_svm)
    if label_svm(i) == actual_label(i)
        accuracy = accuracy + 1;
    end
end
accuracy = accuracy/num_test;
fprintf('Accuracy for test 1 using SVM is %.1f \n', accuracy);
%
test1 = strings(length(random_index),5);
for i = 1:length(random_index)
    test1(i,1) = random_index(i);
    test1(i,2) = song_list(random_index(i));
    test1(i,3) = label(i);
    test1(i,4) = label_svm(i);
    test1(i,5) = actual_label(i);
end
fprintf('song   title             KNN      SVM    Actual \n');
test1