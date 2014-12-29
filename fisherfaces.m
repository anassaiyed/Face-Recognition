function[] = fisherfaces(N,d)

load('data1.mat');

% N=5;
% d=30;

vec_img=cell(15,11);

for i=1:15,
    for j=1:11,
        vec_img{i,j}=reshape(img{i,j},1,1833);
    end
end

train_img=zeros(15*N,1833);
for i=1:15,
    for j=1:N,
        train_img(((i-1)*N)+j,:)=vec_img{i,j}(1,:);
    end
end

tr_mean=zeros(15,1833);
for i=1:15,
    tr_mean(i,:)=mean(train_img(((i-1)*N)+1:((i-1)*N)+N,:));
end

z=zeros(15*N,1833);
for i=1:15*N,
    z(i,:)=train_img(i,:)-tr_mean(ceil(i/N),:);
end

Sw=zeros(1833,1833);
for i=1:15,
    Sw=Sw+(z(((i-1)*N)+1:((i-1)*N)+N,:)'*z(((i-1)*N)+1:((i-1)*N)+N,:));
end

final_mean=zeros(1,1833);
for i=1:15,
    final_mean=final_mean+tr_mean(i);
end
final_mean=final_mean/15;

SB=zeros(1833,1833);
for i=1:15,
    SB=SB+N*((tr_mean(i,:)-final_mean)'*(tr_mean(i,:)-final_mean));
end

[eig_vec,eig_val]=eig(inv(Sw)*SB);
% [eig_vec,eig_val]=eig(Sw\SB);
w=eig_vec(:,1:d);

y=cell(15,1);

for i=1:15,
    y{i}=w'*train_img(((i-1)*N)+1:((i-1)*N)+N,:)';
    y{i}=y{i}';
end


test_img=zeros(15*(11-N),1833);
for i=1:15,
    for j=N+1:11,
        test_img(((i-1)*(11-N))+(j-N),:)=vec_img{i,j}(1,:);
    end
end

test_y=cell(15,1);
for i=1:15,
    test_y{i}=w'*test_img(((i-1)*(11-N))+1:((i-1)*(11-N))+(11-N),:)';
    test_y{i}=test_y{i}';
end

dist=cell(15,1);
for i=1:15,
    dist{i}=zeros(N,1);
end
min_dist=zeros(15,1);
err_cnt=zeros(15,1);
for l=1:15,
    for j=1:(11-N),
        for i=1:N,
            for k=1:15,
                dist{k}(i)=sqrt(sum((y{k}(i,:)-test_y{l}(j,:)).^2));
            end
        end
        for k=1:15,
            min_dist(k,1)=min(dist{k});
        end
        if min(min_dist)~=min_dist(l,1),
            err_cnt(l)=err_cnt(l)+1;
        end
    end
end

error=100*(sum(err_cnt)/(15*(11-N)));
fprintf('\nThe error rate for N=%d and d=%d in Fisherfaces is: %f\n\n',N,d,error);
