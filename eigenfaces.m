function[] = eigenfaces(N,d)

load('data.mat');

% N=7;
% d=10;

vec_img=cell(15,11);

for i=1:15,
    for j=1:11,
        vec_img{i,j}=reshape(img{i,j},1,45045);
    end
end

train_img=zeros(15*N,45045);
for i=1:15,
    for j=1:N,
        train_img(((i-1)*N)+j,:)=vec_img{i,j}(1,:);
    end
end

tr_mean=mean(train_img);

z=zeros(15*N,45045);
for i=1:15*N,
    z(i,:)=train_img(i,:)-tr_mean;
end

scatter=z*z';
[eig_vec,eig_val]=svd(scatter);
final_eig_vec=eig_vec*z;

M=final_eig_vec(1:d,:);

w=M*z';

test_img=zeros(15*(11-N),45045);
for i=1:15,
    for j=N+1:11,
        test_img(((i-1)*(11-N))+(j-N),:)=vec_img{i,j}(1,:);
    end
end

p=zeros(15*(11-N),45045);
for i=1:15*(11-N),
    p(i,:)=test_img(i,:)-tr_mean;
end

w_test=M*p';

w=w';
w_test=w_test';
dist=zeros(15*N,1);
category=zeros(15*(11-N),1);
for i=1:15*(11-N),
    for j=1:15*N,
        dist_temp=w_test(i,:)-w(j,:);
        dist_temp=dist_temp.^2;
        temp=sum(dist_temp);
        dist(j)=sqrt(temp);
    end
    [val,index]=min(dist);
    category(i)=ceil(index/N);
end

hit=0;
for i=1:15*(11-N),
    if category(i)==ceil(i/(11-N)),
        hit=hit+1;
    end
end

error_rate=100-(100*hit/(15*(11-N)));
fprintf('\nThe error rate for N=%d and d=%d in Eigenfaces is: %f\n\n',N,d,error_rate);
