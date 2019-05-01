clear
close all;
clc

% 按文件夹读取数据,数据保存到allData中
db = dir('orl_faces');
face = cell(1, 1); 
allData = zeros(2, 2);
labelList = zeros(1, 1);

acc_index = 0;
for train_data_nums = 6:-1:2
    num = 0;
    type = 0;
    labelMap = containers.Map(0,0);
    for i = 4 : length(db)
        fi = dir([db(i).folder '/' db(i).name]);
        type = type + 1;
        for j = 3 : length(fi)-train_data_nums
            num = num + 1;
            face{i - 2, j - 2} = imread([fi(j).folder '/' fi(j).name]);
            labelMap(num) = type;
            if num == 1
                [imageLen, imageWid] = size(face{i - 2, j - 2});
            end
            allData(1: imageLen * imageWid, num) = double(reshape(face{i - 2, j - 2}, [imageLen * imageWid, 1]));
            % imshow(face{i - 2, j - 2});
        end
        labelList(1, type) = i-3;
    end
    for num = 1:20
        subplot(4, 5, num);imshow(reshape(allData(:, num), [imageLen, imageWid]));
    end
    
    % 求均值向量
    [allDataRows, allDataCols] = size(allData);
    % avgX = 1.0/allDataCols * sum(allData, 2);

    avgX = mean(allData, 2);
    % plot(avgX);
    % imshow(reshape(avgX, [imageLen, imageWid]));

    for num = 1:allDataCols
        allData(:, num) = allData(:, num)-avgX;
    end

    % for num = 1:20
    %     subplot(4, 5, num);imshow(reshape(allData(:, num), [imageLen, imageWid]));
    % end

    % 求协方差矩阵
    % covC = 1.0/allDataCols * C * C';
    covC = allData' * allData;

    % covC = cov(allData);
    [coeffC, latentC, explainedC] = pcacov(covC);

    % 选取构成能量95%的特征值
    numOfLatent = 1;
    proportion = 0;
    while(proportion < 95)
        proportion = proportion + explainedC(numOfLatent);
        numOfLatent = numOfLatent+1;
    end
    numOfLatent = numOfLatent - 1;

    % 求特征脸
    W = allData*coeffC;
    W = W(:, 1:numOfLatent);
    % plot(W(:), 1);

    % for k = 1:20
    %     subplot(4, 5, k);imshow(reshape(W(:, k), [imageLen, imageWid]));
    % end

    % 将训练数据投影到该新的特征空间中
    reference = W' * allData;
    % plot(reference);

    % 测试训练结果准确率
    distances = zeros(1, 1);
    testNum = 0;
    type_index = 0;
    accCnt = 0;
    for i = 4 : length(db)
        fi = dir([db(i).folder '/' db(i).name]);
        type_index = type_index + 1;
        for j = length(fi)-1 : length(fi)
            testNum = testNum + 1;
            imgTest = imread([fi(j).folder '/' fi(j).name]);
            imgTest = reshape(imgTest, [imageLen * imageWid, 1]);
            imgTest = double(imgTest(:));
            imgTest = W' * (imgTest - avgX);

            dis = realmax('double');
            for k = 1:allDataCols
                temp = norm(imgTest - reference(:, k));
                distances(:, k) = norm(imgTest - reference(:, k));
                if(dis > temp)
                    aimOne = k;
                    dis = temp;
                end
            end

            if labelMap(aimOne) == type_index
                accCnt = accCnt + 1;
            end
        end
    end

    % plot(distances);
    acc_index = acc_index + 1;
    accuray(acc_index+3) = accCnt/testNum;
end

% plot(accuray);