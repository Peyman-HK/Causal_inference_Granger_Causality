%% Globally Improving KNN by RVM
%%%%%%%%%%%%%%% ****In CODE doroste***
clear all; close all; clc;   % Clear all workspaces 
LASTN = maxNumCompThreads(str2num(getenv('SLURM_JOB_CPUS_PER_NODE')));
load('/home/peymanhk/k-RV/Data_all_Filtered.mat') 

%%  Define parameters
p = 1; % time lag or model order
Num_ROIs           =     size(Data_all_Filtered(1).img_time_serie,1);
Num_Time_points    =     size(Data_all_Filtered(1).img_time_serie,2);

%% Learning phase - MP
Poly_order = 1:5;
for Num_Subj = 301 : 318
    Num_Subj
    Fij_MP   =  zeros(264,264,size(Poly_order,2));
    Time_series  =  Data_all_Filtered(Num_Subj).img_time_serie;
    for row_ind1  =  1 : Num_ROIs
        % SIMPLE GC
        Bold_i            =  Time_series(row_ind1,:);
        All_Simp          =  Bold_i(1,p:end-1)';
        Y                 =  Bold_i(1,p+1:end)';
        Y_hat_simp        =  zeros(size(Y,1),1);
        Y_hat_full        =  zeros(size(Y,1),1);
        for Num_order = 1 : size(Poly_order,2)  
            PHI_Trd             =  Full_Polynomial(Poly_order(Num_order), All_Simp);
            Y                   =  Bold_i(1,p+1:end)';
            Beta_simp           =  pinv(PHI_Trd)*Y;
            Y_hat_simp          =  PHI_Trd*Beta_simp;
            sigsimp(Num_order)  =  (Y-Y_hat_simp)'*(Y-Y_hat_simp);
        end        
        for row_ind2 = 1: Num_ROIs
            if row_ind2 == row_ind1
                Fij_MP(row_ind1, row_ind2, :,:) = 0;
            else
                Bold_j       =  Time_series(row_ind2,:);
                Tr_X_full    =  [All_Simp'; Bold_j(1,p:end-1)]';
                for Num_order = 1 : size(Poly_order,2)
                    PHI_Trd_full         =  Full_Polynomial(Poly_order(Num_order), Tr_X_full);
                    Beta_full            =  pinv(PHI_Trd_full)*Y;
                    Y_hat_full	         =  PHI_Trd_full*Beta_full;
                    sigfull(Num_order)   =  (Y-Y_hat_full)'*(Y-Y_hat_full);
                end
                Temp_Fij_MP             =  log(sigsimp./sigfull);
                for Num_order = 1 : size(Poly_order,2)
                    Fij_MP(row_ind2,row_ind1,Num_order)  =  Temp_Fij_MP(1,Num_order);
                end
            end
        end
    end
    Causal_Map_MP7{Num_Subj,1} = Fij_MP;
end

save('Causal_Map_MP7.mat','Causal_Map_MP7');
