clear; close all; clc;
%% 1. 参数设置
dataDir = 'C:\Users\33251\Desktop\科研\数据集\初始\直连\waveform_move\'; 
saveDir = 'C:\Users\33251\Desktop\科研\数据集\初始\直连\3\'; 
K_Values =  [4,7,8,9,10,11,12,13,15,16,17,18,40,49,50,51,52,53,54,55,56,57,58,59,60,61,65,66,67,68];
Projection_Bins = 128; 
% 坐标范围 
% I/Q 轴
x_limits = [-0.1, 2.3];     
y_limits = [-0.3, 0.3];     
x_edges = linspace(x_limits(1), x_limits(2), Projection_Bins + 1);
y_edges = linspace(y_limits(1), y_limits(2), Projection_Bins + 1);
% 45度轴 (I+Q)/sqrt(2)
p45_limits = [-0.5, 2.0]; 
p45_edges = linspace(p45_limits(1), p45_limits(2), Projection_Bins + 1);
% 信号处理参数
L = 1;                      
min_density_threshold = 5;  
Nbins_Cleaning = [100 100]; 
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

first_image_path = ''; 

%% 2. 遍历处理
files = dir(fullfile(dataDir, '*.mat'));
fprintf('开始生成 SPF 矩阵，共 %d 个文件...\n', length(files));
for i = 1:length(files)
    fileName = files(i).name;
    fullFilePath = fullfile(dataDir, fileName);
    [~, nameBody, ~] = fileparts(fileName);
    
     deviceImgDir = fullfile(saveDir, 'Images', nameBody);
     if ~exist(deviceImgDir, 'dir')
         mkdir(deviceImgDir);
     end
    
    try
        data_struct = load(fullFilePath);
        if ~isfield(data_struct, 'expendDMrs')
            continue;
        end
        expendDMrs = data_struct.expendDMrs; 
    catch
        continue;
    end
    
    [totalRows, ~] = size(expendDMrs);
    
    Num_K = length(K_Values);
    Feature_Width = Projection_Bins * 3; 
    
    Device_Matrix_Data = zeros(totalRows, Num_K, Feature_Width, 'single'); 
    
    valid_count = 0;
    fprintf('正在处理: %s ... ', nameBody);
    
    for row_idx = 1:totalRows
        one_line_data = expendDMrs(row_idx, :);
        dmrs_interp = interp(one_line_data, L);
        dmrs_expanded = repmat(dmrs_interp, 1, 10);
        
        Current_Feature_Map = zeros(Num_K, Feature_Width);
        is_sample_valid = true;
        
        for k_idx = 1:Num_K
            k = K_Values(k_idx); 
            
            % 1. 差分
            try
                Data_IQ_Offset = F_Data_IQ_Offset_Process(dmrs_expanded', 0);
                Data_Process_Out = F_Differential_Process(Data_IQ_Offset, k);
            catch
                is_sample_valid = false;
                break;
            end
            
            % 2. 清洗
            I_temp = real(Data_Process_Out);
            Q_temp = imag(Data_Process_Out);
            [counts_clean, ~, ~, binX, binY] = histcounts2(I_temp, Q_temp, Nbins_Cleaning);
            point_counts = zeros(size(Data_Process_Out));
            valid_idx = binX > 0 & binY > 0 & binX <= size(counts_clean,1) & binY <= size(counts_clean,2);
            if any(valid_idx)
                idx = sub2ind(size(counts_clean), binX(valid_idx), binY(valid_idx));
                point_counts(valid_idx) = counts_clean(idx);
            end
            Data_Cleaned = Data_Process_Out(point_counts >= min_density_threshold);
            
            if isempty(Data_Cleaned)
                continue; 
            end
            
            % 3. 归一化
            scale = sqrt(mean(abs(Data_Cleaned).^2)); 
            if scale > 0, Data_Cleaned = Data_Cleaned / scale; end
            
            % 4. 三轴投影 (I, Q, 45度)
            I_vals = real(Data_Cleaned);
            Q_vals = imag(Data_Cleaned);
            
            % (A) I轴 (0度)
            Proj_I = histcounts(I_vals, x_edges);
            % (B) Q轴 (90度)
            Proj_Q = histcounts(Q_vals, y_edges);
            % (C) 45度轴
            Rotated_Vals = I_vals * 0.7071 + Q_vals * 0.7071;
            Proj_45 = histcounts(Rotated_Vals, p45_edges);
            
            % 5. 归一化投影
            Proj_I = Proj_I / (max(Proj_I) + 1e-6);
            Proj_Q = Proj_Q / (max(Proj_Q) + 1e-6);
            Proj_45 = Proj_45 / (max(Proj_45) + 1e-6);
            
            % 6. 存入矩阵第 k 行 
            Current_Feature_Map(k_idx, :) = [Proj_I, Proj_Q, Proj_45];
        end
        
        if ~is_sample_valid
            continue;
        end
        
        valid_count = valid_count + 1;
        Device_Matrix_Data(valid_count, :, :) = Current_Feature_Map;
        
         img_to_show = imresize(Current_Feature_Map, [200, 384], 'nearest'); 
         imgName = sprintf('%d.png', row_idx);
         
         current_img_full_path = fullfile(deviceImgDir, imgName);
         imwrite(uint8(img_to_show * 255), current_img_full_path);

         if isempty(first_image_path)
             first_image_path = current_img_full_path;
         end
    end
    
    Device_Matrix_Data = Device_Matrix_Data(1:valid_count, :, :);
    
    % 保存数据
    saveName = fullfile(saveDir, [nameBody '.mat']);
    save(saveName, 'Device_Matrix_Data', 'K_Values');
    
    fprintf('完成 (样本数: %d)\n', valid_count);
end
fprintf('所有处理完毕。数据和图保存在: %s\n', saveDir);

if ~isempty(first_image_path) && exist(first_image_path, 'file')
    figure('Name', '处理完成 ', 'NumberTitle', 'off', 'Position', [300, 200, 800, 600]);
    imshow(first_image_path, 'InitialMagnification', 'fit');
    title('生成的DCPT特征图','FontSize', 16);
end