% Enter type of classification as text
% Option: band, genre, random

function [training_set, info_matrix, song_list, artist]= getTrainingSet(folder_name)
    trainingSet_dir = dir(folder_name);
    trainingSet_dir = trainingSet_dir(3:end);
    sbf = length(trainingSet_dir); % sbf = size_big_folder
    info_matrix = strings(sbf, 4);
    num_song = 0;
    for i = 1: sbf
        sub_folder = dir([folder_name '/' trainingSet_dir(i).name]);
        num_song = num_song + length(sub_folder(3:end));
        info_matrix(i,1)= trainingSet_dir(i).name;
        info_matrix(i,2) = num2str(length(sub_folder(3:end)));
        info_matrix(i,4) = num_song;
    end
    
    sample = 240000; %sampling each song at specific time 5s
    info_matrix(:,3) = sample;
    training_set = zeros(sample, num_song);
    song_list = strings(num_song,1);
    artist = song_list;
    index = 0;  
    for i = 1:sbf
        sub_folder = dir([folder_name '/' trainingSet_dir(i).name]);
        for j = 1: length(sub_folder(3:end))   
            index = index + 1;
            [x,~] = audioread([folder_name '/' trainingSet_dir(i).name '/' sub_folder(j+2).name]);
            new_song = zeros(sample,1);
            for k = 1:sample
                new_song(k,1) = x(2*k-1,1);              
            end
            training_set(:,index) = new_song;
            song_list(index,1) = sub_folder(j+2).name;
            artist(index,1) = trainingSet_dir(i).name;
        end
    end
end