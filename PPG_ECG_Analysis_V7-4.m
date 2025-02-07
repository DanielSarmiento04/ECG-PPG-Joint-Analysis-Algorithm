%/**
%@author Diana Carolina Martínez Reyes (dmare)
%This algorithm reads the values corresponding to systolic and diastolic
%blood pressure from the patient dataset. As well it reads the
%electrocardiogram (ECG) and photoplethysmogram (PPG) signal values. Then it procedess to
%analize the electrocardiogram signal to identify the heart rate and heart
%rate variability. The process continues with the ECG and PPG signal processing and analisis and ends with the blood pressure estimation and validation. 
%*/
Data=load('DataSet_ECG_PPG_p2506.txt');
NLines=length(Data);

SystolicPressure=zeros(1,40);
DiastolicPressure=zeros(1,40);
heart_rate=zeros(1,40);
t0=zeros(1,40);
ta=zeros(1,40);
tms=zeros(1,40);
tsd1=zeros(1,40);
Ra=zeros(1,40);  
t1=zeros(1,40);
tdb=zeros(1,40);
for y = 1
close all
RecNumb=y; %To choose which sample want to be analized
%Bringing signal from Database to Matlab WorkSpace
PatientID=Data(RecNumb,1);
ECGNumb=Data(RecNumb,2);
ECGName=[num2str(ECGNumb) '-1K-raw.wav']; 
PPGNumb=Data(RecNumb,3);
PPGName=[num2str(PPGNumb) '-ppg.wav']; 
SystolicPressure(y)=Data(RecNumb,4);
DiastolicPressure(y)=Data(RecNumb,5);
                                           %%%%%ECG%%%%%

PathECG='ECG';
%OpenECG=['.\' PathECG '\' ECGName];
OpenECG=['E:\OneDrive - UNIVERSIDAD INDUSTRIAL DE SANTANDER\Master\Research Project\4 Semester\ECG\56677-1K-raw.wav']; 
[ECG, Fsecg] = audioread(OpenECG);
LECG=length(ECG(:,1));
Tecg=1/Fsecg; %Sample time ECG
tecg=0:Tecg:(LECG-1)*Tecg; %Signal time vector
figure(1)
plot(tecg,ECG)
set(gca,'XLim',[0 tecg(LECG)])
title('ECG')
xlabel('Time')
ylabel('Amplitude')
inps=25*Fsecg;
finps=85*Fsecg;
ECG_w=ECG(inps:finps,:); %50 seconds of the signal 
LECG_w=length(ECG_w(:,1));
tecg_w=0:Tecg:(LECG_w-1)*Tecg; %Signal time vector
%%%
%Finding Syncronization Points corresponding to R-R interval 
%%%From external program
PathECG='ECG';
RR_intervalsName=[num2str(ECGNumb) '-1K-raw.RR']; 
%OpenRR_intervals=['.\' PathECG '\' RR_intervalsName];
OpenRR_intervals=['E:\OneDrive - UNIVERSIDAD INDUSTRIAL DE SANTANDER\Master\Research Project\4 Semester\ECG\56677-1K-raw.RR'];
RR_intervals_data=load(OpenRR_intervals,'-ascii'); 
RR_intervals=RR_intervals_data(:,2);
RR_intervals_mean=mean(RR_intervals);
RR_intervals_SD=std(RR_intervals); %to evaluate how much is changing the interval 
%%%RPositions
R_prev_position=0;
RRPositions=zeros(1,length(RR_intervals));
for i=1:length(RR_intervals)
    RRPositions(i)=RR_intervals(i)+R_prev_position;
    R_prev_position=RRPositions(i); 
end 
RnewPositions=0;
stp=find(RRPositions>=inps);
stp=stp(1); 
RPositionstp=RRPositions(stp);
ftp=find(RRPositions<=finps);
ftp=ftp(end); 
RPositionftp=RRPositions(ftp);
RPositions=RRPositions(stp:ftp);
RPositions=RPositions-inps;
RR_interval=round(mean(RR_intervals)); 
figure(2)
plot(tecg,ECG)
xlabel('Time')
ylabel('Amplitude')
title('ECG')
set(gca,'XLim',[0 tecg(LECG)])
hold on 
RAmp=(ECG(RRPositions))'; 
RAmp_mean=mean(RAmp); 
tRpeak=RRPositions*Tecg; 
plot(tRpeak,RAmp,'*');
%title('ECG with R Points')
xlabel('Time')
ylabel('Amplitude')
hold off
HR=round((60*Fsecg)/RR_interval); 
heart_rate(y)=HR;   
%%%
%%Wavelet Discrete Analysis%%%
%%Wavelet decomposition
%%%Continuous Wavelet Analysis
% figure(50)
%  cwt(ECG_w,Fsecg); 
[WT,F]=cwt(ECG_w,Fsecg);
if HR<=54
    fint1=8;
    ffin1=28;
else
    if  HR<=62
        fint1=8;
        ffin1=28;
    else
        if HR<=72
            fint1=8;
            ffin1=28;
        else
        if HR<=82
            fint1=8;
            ffin1=28;
        else
            fint1=8;
            ffin1=28;   
        end
        end
    end 
end
%ECG_filtered=icwt(WT,F,[fint1 ffin1],'SignalMean',mean(ECG_w), 'Wavelet', 'morl'); %8 to 32
%ECG_filtered = icwt(WT, F, [fint1 ffin1], 'Wavelet', 'morl');
ECG_filtered = icwt(WT);
ECG_filtered=ECG_filtered'; 
figure(4)
subplot(2,1,1)
plot(tecg_w,ECG_filtered)
title('ECG after Wavelet Analysis')
xlabel('Time')
ylabel('Amplitude')
subplot(2,1,2)
plot(tecg_w,ECG_w)
title('Raw ECG')
xlabel('Time')
ylabel('Amplitude')
%%%
                                                %%%%PPG%%%%%
PathPPG='PPG'; %Path to the PPG folder
OpenPPG=['.\' PathPPG '\' PPGName];
[PPG, Fs] = audioread(OpenPPG);
LPPG=length(PPG(:,1)); %Length of signal (Number of rowns)
CPPG=length(PPG(1,:)); %Number of columns 
T=1/Fs; %Sample time
tppg=0:T:(LPPG-1)*T; %Time vector
figure(1)
subplot(2,1,2)
plot(tppg,PPG)
set(gca,'XLim',[0 tppg(LPPG)])
title('PPG')
PPG_w=PPG(inps:finps,:); %50 seconds of the signal 
LPPG_w=length(PPG_w(:,1)); %Length of signal (Number of rowns)
tppg_w=0:T:(LPPG_w-1)*T; %Time vector
 
%%%
%%Continuous Wavelet Analysis
% figure(50)
% cwt(PPG_w,Fs)
%[WT,F]=cwt(PPG_w,Fs);
[WT, F] = cwt(PPG_w, 'bump');
 if HR<=50
    fint=1.5;
    ffin=4;
else
if HR<=54
    fint=1.5;
    ffin=4;
else
    if  HR<=62
        fint=1.8;
        ffin=4;
    else
        if HR<=72
            fint=2;
            ffin=6;
        else
        if HR<=82
            fint=2;
            ffin=6;
        else
            fint=2;
            ffin=6;   
        end
        end
    end 
end
 end
%PPG_Wave_Filtered=icwt(WT,F,[fint ffin],'SignalMean',mean(PPG_w));
PPG_Wave_Filtered=icwt(WT, 'bump');
figure(3)
subplot(2,1,1)
plot(tppg_w,PPG_Wave_Filtered)
title('PPG After Wavelet Analysis')
xlabel('Time')
ylabel('Amplitude')
subplot(2,1,2)
plot(tppg_w,PPG_w)
title('Raw PPG')  
figure(100)
plot(tppg_w,PPG_Wave_Filtered,'LineWidth',1.5)
xlabel('Time','FontSize',14)
ylabel('Amplitude','FontSize',14)
title('PPG','FontSize',14) 
%To find the waveform model
%%%Reshaping conform RR cycles%%%
M=mod(LPPG_w,RR_interval); %Remainder after division 
LPPG2=LPPG_w-M; %To make sure the divLPPG); %Changing the length of the signal
a=LPPG2/RR_interval; %Number of subinterval for PPG according to syincro points in ECG 
PPG2=PPG_Wave_Filtered(1:LPPG2); 
tppg2=0:T:(LPPG2-1)*T;
PPG_2_1=reshape(PPG2,RR_interval,a); %Dividing PPG signal into different subgroups corresponding one period signal aprox
PPG_2_1=PPG_2_1'; 
LPPG_2=length(PPG_2_1(:,1));
PPG_Wave_Filtered=PPG_Wave_Filtered'; 
increment=round(RR_interval*0.03);
Llwave=(RR_interval+(20*increment)+1);
PPG_2=zeros((LPPG_2-1),Llwave); 
PPG_2(1,:)=PPG_Wave_Filtered(1:(RR_interval+(20*increment)+1)); 
 for i=2:(LPPG_2-1)
   initial=RR_interval*(i-1); 
   final=initial+RR_interval;
   %ab=initial-(10*increment);
   ab=initial;
   bc=(final+(20*increment));
   PPG_2(i,:)=PPG_Wave_Filtered(ab:bc); 
 end
Number_waves=length(PPG_2(:,1)); %Number of subgroups 
Number_samples=length(PPG_2(1,:));  %Number of elements in each subgroup
PPG_2det=PPG_2; 
%%%Correlation Coeficient%%%
Matrix_CorCoe=zeros(Number_waves,Number_waves);
for i = 1:Number_waves
        for j = 1:Number_waves
            CCoe=corrcoef(PPG_2det(i,:) , PPG_2det(j,:)); 
        Matrix_CorCoe(i,j)=CCoe(1,2); %Matrix of CorrCoef Between subgroups 
        end  
end
%%%Looking for Wave models with high Correletaion Coefficient%%%
Matrix_CorCoe(Matrix_CorCoe>0.999)=0; %Cause some values are 1.0000 and is a problem later
max_Matrix_CorCoe=max(Matrix_CorCoe); 
[I,J]=find(Matrix_CorCoe>0.98); %Looking for Waves which CorrCoef>0.989
if isempty(I)== 1
    [I,J]=find(Matrix_CorCoe>0.96); %Looking for Waves which CorrCoef>0.989
end
MccRows =length(Matrix_CorCoe(:,1)); %Number of Rows matrix correlation coefficients
MccColumns =length(Matrix_CorCoe(1,:)); %Number of Rows matrix correlation coefficient
PPG_Models=PPG_2det(I,:); %Matrix with waves with correlation coefficient higher than threshold 
figure(5) 
Lmodels=length(PPG_Models(1,:)); 
tmodels=0:T:(Lmodels-1)*T;
plot(tmodels,PPG_Models,'LineWidth',1.5)
title('PPG Models','FontSize',14)
xlabel('Time','FontSize',14)
ylabel('Amplitude','FontSize',14)


Mod=0; %Initial Value for model Sum of Correlation coefficients respect it 
Sum_Max=zeros(1,length(I)); %Vector sum of corrcoef values most similar waves respect model(i)
 
 for i = 1:length(I) %To do it with all the best models 
     Pos_Local=I(i); %Number of model wave
Max_Local=Matrix_CorCoe(Pos_Local,:); %Only corrcoef respect model
%%%To divide the signal into subgroups%%%
blockSizeX = 6; %Height of the subarray to divide in groups
numFullSizeBlocksX = floor(MccColumns / blockSizeX); %An integer value of the division
blockHeights = blockSizeX * ones(numFullSizeBlocksX, 1);
partialBlockX = rem(MccColumns, blockSizeX); % Find out if there is a remaining smaller block that didn't fit the size
if partialBlockX ~= 0
     blockHeights = [blockHeights; partialBlockX]; % If there is a smaller block, add it on to the block size array
end
 Matrix_CorCoe_Local= mat2cell(Max_Local,1,blockHeights); %Corrcoef respect model clustered in small groups 
 LMC_Local=length(Matrix_CorCoe_Local); %Number of subgroups 
 Max_CorrCoef=zeros(1,LMC_Local); %Vector containing max corrcoef each subgroup with respect to model
 Loc_Max_CorrCoef=zeros(1,LMC_Local);  %Local Location of those maximums 
 for j= 1:LMC_Local %To evaluate each subgroup 
      a=cell2mat(Matrix_CorCoe_Local(j)); %Each submatrix
      [Max_sub,Loc_Max_Sub]=max(a); %Find the max corrcoed for each subgroup
      if Max_sub>0.9
      Max_CorrCoef(j)=Max_sub; %Vector containing max corrcoef with respect to model
      Loc_Max_CorrCoef(j)=Loc_Max_Sub; %Location of those maximums (In each subgroup)
      end
 end
Max_CorrCoef(Max_CorrCoef==0)=[]; 
Sum_Max(i)=sum(Max_CorrCoef); %The sum of all max corrcoef to compare with different models
 if Sum_Max(i)>=Mod
     Mod=Sum_Max(i);
     Model_Position=I(i); %which is the model number 
     Model=PPG_2(Model_Position,:); %Wave corresponding to model
   Loc_BestModels=Loc_Max_CorrCoef; %Location of waves most similar to model
     %%%To convert Local position to Global%%%
 Loc_Best_Waves=zeros(1,length(Loc_BestModels) ); %Array containing location of most similar waves
 if Loc_BestModels(1)~=0
 Loc_Best_Waves(1)=Loc_BestModels(1); %First position donot change
 end
 for k=1:(length(Loc_BestModels)-1)
     if Loc_BestModels(k+1)~=0
        LL=blockSizeX*k; %Coverting local position to global
        Loc_Best_Waves(k+1)=Loc_BestModels(k+1)+LL; %Array with the global position of most similar waves
     end
 end 
 Loc_Best_Waves(Loc_Best_Waves==0)=[]; 
 LLoc_Best_Waves = length(Loc_Best_Waves); 
 Group_Models=PPG_2(Loc_Best_Waves,:); %Matrix containing models waves 
 end 
 numberRowModels=length(Group_Models(:,1));
 end
 if numberRowModels>1
             medn=round(LLoc_Best_Waves/2);
             PPGmodel_loc=Loc_Best_Waves(medn); %%%Wave which is located in the middle
             PPGmodel=PPG_2(PPGmodel_loc,:);
 else
     PPGmodel=Group_Models; 
 end
figure(6)
plot(tmodels,Group_Models)
title('PPG Best Waves') 
Lmodel=length(PPGmodel(1,:)); 
tmodel=0:T:(Lmodel-1)*T;
figure(7);
plot(tmodel,PPGmodel,'LineWidth',1.5)
title('PPG Single Cycle Waveform Model','FontSize',14)
xlabel('Time','FontSize',14)
ylabel('Amplitude','FontSize',14)
%saveas(h,sprintf('PPGModel%d.png',y));
%%%
%%%Wave Features Extraction%%%
                                            %%%Blood Pressure%%%
%%%To analyze and obtain meaningful points 
%%%PPG delay respect ECG 
        %%%ECG
init=(RPositions(PPGmodel_loc-1))+60;       
if init<=1
initial=0;     
else  
initial=init;
end
if (PPGmodel_loc+1)<=length(RPositions)
finl=(RPositions(PPGmodel_loc+1))-40;
else
finl=LPPG_w;  
end
final=finl;
new_ECGmodel=ECG_filtered(initial:final); 
Lnew_ECGmodel=length(new_ECGmodel);
tnew_ECGmodel=0:Tecg:(Lnew_ECGmodel-1)*Tecg;
figure(30)
plot(tnew_ECGmodel,new_ECGmodel)
hold on
% ECG_dif=diff(new_ECGmodel); %Obtaining the first derivate for the signal
% plot(ECG_dif)
% hold off 
 %%%Looking for R-peak
[Rpeak,RLoc]=max(new_ECGmodel); 
tRpeak=RLoc*Tecg;
plot(tRpeak,Rpeak,'*')
h=figure;
figure(8)
subplot(3,1,1)
Lnew_ECGmodel=length(new_ECGmodel); 
tnew_ECGmodel=0:Tecg:(Lnew_ECGmodel-1)*Tecg;
plot(tnew_ECGmodel,new_ECGmodel,'b','LineWidth',1.5) 
xlim([0 (Lnew_ECGmodel-1)*Tecg])
xlabel('Time','FontSize',12)
ylabel('Amplitude','FontSize',12)
title('ECG') 
hold on 
plot(tRpeak,Rpeak,'*')
hold off
% % figure(120)
% % initial2=RPositions(PPGmodel_loc)-(increFact*5);
% % final2=RPositions(PPGmodel_loc)+RR_interval-(2*increFact);
% % PPGmodelf2=PPG_Wave_Filtered(initial2:final2);
% % LPPGmodelf2=length(PPGmodelf2);
% % t_PPGmodelf2=0:T:((LPPGmodelf2)-1)*T;
% % subplot(2,1,1)
% % plot(t_PPGmodelf2,PPGmodelf2,'b','LineWidth',1.5) 
% % xlim([0 (LPPGmodelf2-1)*T])
% % xlabel('Time','FontSize',12)
% % ylabel('Amplitude','FontSize',12)
% % title('PPG model')
% % PPGmodelf2_velocity=diff(PPGmodelf2); 
% % LPPGmodelf2_velocity=length(PPGmodelf2_velocity);
% % t_PPGmodelf2_velocity=0:T:((LPPGmodelf2_velocity)-1)*T;
% % subplot(2,1,2)
% % plot(t_PPGmodelf2_velocity,PPGmodelf2_velocity,'b','LineWidth',1.5) 
% % xlim([0 (LPPGmodelf2_velocity-1)*T])
% % xlabel('Time','FontSize',12)
% % ylabel('Amplitude','FontSize',12)
% % title('PPG first derivative')
% %%%%
        %%%%PPG
new_PPGmodel_Filtered=PPG_Wave_Filtered(initial:final);
new_PPGmodel_velocity=diff(new_PPGmodel_Filtered); 
Lnew_PPGmodel_velocity=length(new_PPGmodel_velocity);
t_new_PPGmodel_velocity=0:T:((Lnew_PPGmodel_velocity)-1)*T;
%Systolic
[SystolicPeak,LSystolicPeak_Position] =max(new_PPGmodel_Filtered(RLoc:end)); 
SystolicPeak_Position=LSystolicPeak_Position+RLoc;
tSystolicPeak=(SystolicPeak_Position)*T;
%Baseline
PPG_Acceleration=diff(new_PPGmodel_Filtered,2);
[Baseline2,LBaseline2_Position]=min(new_PPGmodel_Filtered(RLoc:SystolicPeak_Position)); 
Baseline2_Position=LBaseline2_Position+RLoc;
t_Baseline2=(Baseline2_Position)*T;
%Maximum slope point 
[maxSlSys_v,LmaxSlSys_Position] =max(new_PPGmodel_velocity(RLoc:SystolicPeak_Position)); 
maxSlSys_Position=LmaxSlSys_Position+RLoc;
tmaxSlSys_Position=(maxSlSys_Position)*T; 
maxSlope=new_PPGmodel_Filtered(maxSlSys_Position); 
Lnew_PPGmodel_Filtered=length(new_PPGmodel_Filtered); 
tnew_PPGmodel_Filtered=0:T:(Lnew_PPGmodel_Filtered-1)*T;
subplot(3,1,2)
plot(tnew_PPGmodel_Filtered,new_PPGmodel_Filtered,'g','LineWidth',1.5) 
xlim([0 (Lnew_PPGmodel_Filtered-1)*T])
xlabel('Time','FontSize',12)
ylabel('Amplitude','FontSize',12)
title('PPG')
hold on 
plot(tSystolicPeak,SystolicPeak,'*')
hold on 
plot(t_Baseline2,Baseline2,'*')
hold on 
plot(tmaxSlSys_Position,maxSlope,'*')
hold off
subplot(3,1,3)
plot(t_new_PPGmodel_velocity,new_PPGmodel_velocity,'b','LineWidth',1.5)
xlim([0 (Lnew_PPGmodel_velocity-1)*T])
xlabel('Time','FontSize',12)
ylabel('Amplitude','FontSize',12)
title('PPG First Derivate')
saveas(h,sprintf('FIGPTT%d.png',y));
t0(y)=tSystolicPeak-tRpeak; %delay Systolic point in PPG respect R peak in ECG 
ta(y)=t_Baseline2-tRpeak; %delay Baseline point in PPG respect R peak in ECG 
tms(y)=tmaxSlSys_Position-tRpeak; %delay point maximum  slope or systolic in PPG respect R peak in ECG 
%%%PPG Points %%%
%%%PPG Velocity and Acceleration
if (PPGmodel_loc+1)>length(RPositions)
   final=LPPG_w; 
else
if ((RPositions(PPGmodel_loc+1))+260)>=LPPG_w
final=LPPG_w;     
else   
final=(RPositions(PPGmodel_loc+1))+260;
end
end
new_PPGmodel_Filtered=PPG_Wave_Filtered(initial:final);
tnew_PPGmodel_Filtered=0:T:(length(new_PPGmodel_Filtered)-1)*T;
figure(9); %Signal / Velocity / Acceleration 
subplot(3,1,1)
plot(tnew_PPGmodel_Filtered,new_PPGmodel_Filtered,'r')  
PPG_Velocity=diff(new_PPGmodel_Filtered); 
t_velocity=0:T:(length(PPG_Velocity)-1)*T;
subplot(3,1,2)
plot(t_velocity,PPG_Velocity,'b') 
title('PPG Velocity') 
hold on 
PPG_Velocity_mean=mean(PPG_Velocity);
PPG_Velocity_max=max(PPG_Velocity); 
Dias_Threshold=(PPG_Velocity_max-PPG_Velocity_mean)/2; 
plot(tnew_PPGmodel_Filtered,Dias_Threshold*ones(length(tnew_PPGmodel_Filtered)),'m') 
plot(tnew_PPGmodel_Filtered,0*ones(length(tnew_PPGmodel_Filtered)),'m') 
hold off
PG_Acceleration=diff(new_PPGmodel_Filtered,2);
t_acceleration=0:T:(length(PPG_Acceleration)-1)*T;
subplot(3,1,3)
plot(t_acceleration,PPG_Acceleration,'b') 
hold on
plot(tnew_PPGmodel_Filtered,0*ones(length(tnew_PPGmodel_Filtered)),'m') 
hold off
title('PPG Accelaration') 
%saveas(h,sprintf('FIGVel&Acel%d.png',y));
%%%%Analyzing the signal after systolic peak%%%
PPGmodel=new_PPGmodel_Filtered;
Lmodel9=length(PPGmodel); 
tmodel9=0:T:(Lmodel9-1)*T;
vel_sign=sign(PPG_Velocity);
vel_sign=vel_sign';
%for dicrotic notch 
vel_infle_pt=strfind(vel_sign(SystolicPeak_Position:end),[-1 1]); %find where the signal changes from - to +
vel_infle_pt=vel_infle_pt+SystolicPeak_Position; 
if length(vel_infle_pt)> 1
   vel_infle_pt=vel_infle_pt(1);
 end
Dic1_Position=vel_infle_pt;
t_Dic1=Dic1_Position*T; 
Dic1=new_PPGmodel_Filtered(Dic1_Position);
tdb(y)=t_Dic1-t_Baseline2;
%for diastolic peak 
vel_infle_pt3=strfind(vel_sign((Dic1_Position+50):end),[1 -1]); %find where the signal changes from + to -
vel_infle_pt3=vel_infle_pt3+Dic1_Position+50; 
if length(vel_infle_pt3)> 1
   vel_infle_pt3=vel_infle_pt3(1);
 end
Dic3_Position=vel_infle_pt3;
t_Dic3=Dic3_Position*T; 
Dic3=new_PPGmodel_Filtered(Dic3_Position);
%%%Plot corresponding wave with Baseline/Systolic & Dicrotic points %%%%
h=figure(10); %%%PPG filtered with features identified 
plot(tmodel9,new_PPGmodel_Filtered,'b','LineWidth',1.5) 
xlabel('Time','FontSize',12)
ylabel('Amplitude','FontSize',12)
title('PPG')
xlim([0 (Lmodel9-1)*T])
ylim([min(PPGmodel) max(PPGmodel)])
hold on  
plot(tSystolicPeak,SystolicPeak,'*')
hold on 
plot(t_Baseline2,Baseline2,'*')
hold on 
plot(t_Dic1,Dic1,'*');
hold on  
plot(t_Dic3,Dic3,'*');
saveas(h,sprintf('FIGPPGwf%d.png',y));
tsd1(y)=t_Dic1-tSystolicPeak; %%%Time span between Systolic and dicrotic notch 
Ra(y)=SystolicPeak/Dic3; %%%Ratio between Systolic and dicrotic peaks  
%%%To the line which defines curve
%for inflection Point before systolic point
PPG_Acc=diff(PPGmodel,2);
Acceleration_sign=sign(PPG_Acc); %The sign of the signal acceleration
Acceleration_sign=Acceleration_sign'; 
mRectTang=abs(diff(PPGmodel)./diff(tmodel9)); %%%slope of the tangent line each point of the curve 
infle_point1=strfind(Acceleration_sign(Baseline2_Position:(SystolicPeak_Position-10)),[1 -1]); %find inf point :where acce change sign
Linfle_point1=length(infle_point1); %length of the array to check if found more than one
if Linfle_point1>1
 Slope_infP1=mRectTang(infle_point1); %The slope at inflection point  
[mRectTang_infle_point1,infle_point11]=max(Slope_infP1); %Choose the point with the highest slope
infle_point1=infle_point1(infle_point11); 
end  
infle_pointsPPG=(infle_point1+Baseline2_Position); 
tinfle_pointsPPG=infle_pointsPPG*T; 
infle_points=interp1(tmodel9,PPGmodel,tinfle_pointsPPG,'linear');
plot(tinfle_pointsPPG,infle_points,'ro');
hold on
%%%For the lines which represent the inflection points 
dy=diff(PPGmodel)./diff(tmodel9); %%%slope of the tangent line each point of the curve
Ylines=zeros(length(infle_pointsPPG),length(tmodel9)); 
for i=1:length(infle_pointsPPG)
    infle_pointPPG=infle_pointsPPG(i);
    tinfle_pointsPPG=infle_pointPPG*T; 
    s1=(tmodel9-tinfle_pointsPPG);
    s2=dy(infle_pointPPG);
    s3=PPGmodel(infle_pointPPG);
    tangs=s1*s2+s3;
    plot(tmodel9,tangs,'r')
    scatter(tinfle_pointsPPG,PPGmodel(infle_pointPPG))
    Ylines(i,:)=tangs;    
end
Y_Baseline=Baseline2*ones(1,Lmodel9);
plot(tmodel9,Y_Baseline,'r')
saveas(h,sprintf('FIGPPGwf%d.png',y));
hold off %%%
%%%To find the intersection points between the parameters lines
Base_curve=round(Y_Baseline,3); %Line which defines Baseline
in1_curve=round(Ylines(1,:),3); %Line which defines incresing signal for syastolic
t1o=find(Base_curve==in1_curve); %intersection time Base curve Line1 
if length(t1o)>1
    t1o=mean(t1o);    
end
t1(y)=(t1o-Baseline2_Position)*T; %resultant int point minus Baseline point



end % program 
BloodPressure=[DiastolicPressure ; SystolicPressure]; 
BP_Related_Par=[t0;ta;tms;tsd1;Ra;t1;tdb];
