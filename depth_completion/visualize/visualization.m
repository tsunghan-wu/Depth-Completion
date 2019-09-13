% Visualize depth map in jet color map

% data orginization
% root_path/
% ├── model-A
% │   ├── xxx.mat
% │   └── ...
% └── model-B 

% Set mat data root
root_path = './mat/';
% Models
raw_path = 'raw_depth';
render_path = 'render_depth';
yindaz_path = 'yindaz';
resnet_path = 'resnet';
sa_path = 'sa';
sa_ssim_path = 'sa_ssim';
sa_ssim_bc_path = 'sa_ssim_bc';

paths = {raw_path, render_path, yindaz_path,...
         resnet_path, sa_path, sa_ssim_path,...
         sa_ssim_bc_path};

N = 474;
input_fname = 'filenames.txt';
input_list = textread(input_fname, '%s', N);

save_dir = './visualize_final_ver';

for ii = 1 : N
    ii
    for jj = 1 : numel(paths)
        mat_path = strcat(root_path, paths{jj}, '/', input_list{ii});
        load(mat_path);
    end
    colormap = jet(double(max([raw_depth(:);render_depth(:);yindaz(:);sa(:);sa_ssim(:);sa_ssim_bc(:);resnet(:);])));
    out_dir = split(input_list{ii}, '.');
    out_dir = strcat(save_dir, '/', out_dir{1});
    mkdir(out_dir);
    raw_out_path = strcat(out_dir, '/', 'raw_depth.png');
    render_out_path = strcat(out_dir, '/', 'render_depth.png');
    yindaz_out_path = strcat(out_dir, '/', 'yindaz.png');
    sa_out_path = strcat(out_dir, '/', 'sa.png');
    sa_ssim_out_path = strcat(out_dir, '/', 'sa_ssim.png');
    sa_ssim_bc_out_path = strcat(out_dir, '/', 'sa_ssim_bc.png');
    resnet_out_path = strcat(out_dir, '/', 'resnet.png');
    
    imwrite(label2rgb(raw_depth, colormap), raw_out_path);
    imwrite(label2rgb(render_depth, colormap), render_out_path);
    imwrite(label2rgb(yindaz, colormap), yindaz_out_path);
    imwrite(label2rgb(sa, colormap), sa_out_path);
    imwrite(label2rgb(sa_ssim, colormap), sa_ssim_out_path);
    imwrite(label2rgb(sa_ssim_bc, colormap), sa_ssim_bc_out_path);
    imwrite(label2rgb(resnet, colormap), resnet_out_path);
    
end
