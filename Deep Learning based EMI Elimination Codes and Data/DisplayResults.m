% Display measured/predicted/corrected EMI spectra
% =========================================================================

clear all;close all;clc;

root = '~\Deep Learning based EMI Elimination Codes and Data\';
dir  = [root 'demo\results\'];
sub_file = {'label','output'};
Nfile = size(sub_file,2);

ksize = [128 126 32];

for ii = 1:Nfile
    dir_tmp  = [dir, sub_file{ii}, '.h5'];
    data_tmp = h5read(dir_tmp,'/k-space');
    ksp(:,:,:,ii) = reshape(squeeze(data_tmp(:,:,1,:) + j*data_tmp(:,:,2,:)),ksize);
    clear dir_tmp data_tmp
end

ksp(:,:,:,3) = ksp(:,:,:,1)-ksp(:,:,:,2); % EMI cancellation

fksp = sqrt(size(ksp,1))*fftshift(ifft(ifftshift(ksp,1),[],1),1);
fksp_avg = squeeze(mean(mean(abs(fksp),2),3));

BW = 10;
xx = linspace(-BW/2,BW/2,ksize(1));

figure,plot(xx,fksp_avg);xlim([-BW/2,BW/2]);ylim([0 20]);
       title('Magnitude averaged EMI spectra');grid on;
       ylabel('Magnitude');xlabel('Frequency (kHz)');
       legend('Measured','Predicted','Corrected','location','northwest');legend('boxoff');



