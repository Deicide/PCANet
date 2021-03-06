#include<opencv2/opencv.hpp>
#include<opencv2/core/eigen.hpp>
#include<iostream>
#include<fstream>
#include<stdio.h>
#include<math.h>
#include<ctime>
#include"utils.h"
#include<omp.h>
#include<sys/time.h>
#include<Eigen/Dense>
#include<Eigen/Eigenvalues>
using namespace std;
using namespace cv;
using namespace Eigen;

void out_file(char *var_name,cv::Mat &var)
{
    char filename[100];
    sprintf(filename,"%s.xml",var_name);
    FileStorage fs(filename,FileStorage::WRITE);
    fs<<var_name<<var;
    fs.release();
}

void print_timestamp(char *name)
{
    time_t t;
    struct tm tmval;
    time(&t);
    tmval = *localtime(&t);
    sprintf(name,"%s%d_%d_%d_%d_%d",name,tmval.tm_mon+1,tmval.tm_mday,tmval.tm_hour,tmval.tm_min,tmval.tm_sec);
}

void out_liblinear(char *file_name,cv::Mat &feature,cv::Mat &label)
{
    int length = feature.rows;
    ofstream of(file_name);
    for(int i=0;i<length;i++)
    {
        of<<label.at<float>(i,0)<<' ';
        for(int j=1;j<=feature.cols;j++)
        {
            of<<j<<":"<<feature.at<float>(i,j-1)<<" ";
        }
        of<<'\n';
    }
    of.close();
}
void out_liblinear2(char *file_name,cv::Mat &feature,vector<uchar> &label)
{
    int length = feature.rows;
    ofstream of(file_name);
    for(int i=0;i<length;i++)
    {
        of<<(int)label[i]<<' ';
        for(int j=1;j<=feature.cols;j++)
        {
            of<<j<<":"<<feature.at<float>(i,j-1)<<" ";
        }
        of<<'\n';
    }
    of.close();
}
template <class T>
void out_mat(char *name,cv::Mat &in_mat)
{
    ofstream of(name);
    //for(int row = 0 ; row<in_mat.rows; row++)
    for(int row = 0 ; row<in_mat.rows; row++)
    {
        for(int col = 0; col<in_mat.cols;col++)
        {
            of<<in_mat.at<T>(row,col)<<'\t';
        }
        of<<endl;
    }
    of.close();
}

///convert each patch from the image into a row vector of the output matrix
///THE PATCH MIGHT CROSS OVER THE RANGE OF THE MATRIX!!!!
///the depth of in_img should be 64F
template<class T>
cv::Mat im2col_step(cv::Mat &in_img,int patch_size,int step)
{

    int out_rows = ((in_img.rows - patch_size)/step + 1)*((in_img.cols - patch_size)/step + 1);
    int out_cols = patch_size * patch_size;
    int r_diff = in_img.rows-patch_size;
    int c_diff = in_img.cols-patch_size;
    int out_row_num=0;
    T *p_out,*p_in;
    cv::Mat out_patches(out_rows,out_cols,in_img.depth());

    for(int r_offset=0; r_offset <= r_diff; r_offset+=step)
    {
        for(int c_offset=0; c_offset <= c_diff; c_offset+=step)
        {
            p_out = out_patches.ptr<T>(out_row_num);

            for(int r=0;r<patch_size;r++)
            {
                p_in = in_img.ptr<T>(r+r_offset);
                for(int c=0;c<patch_size;c++)
                {
                    p_out[r*patch_size+c] = p_in[c_offset+c];
                }
            }

            out_row_num++;
        }
    }

    /*FileStorage fs("out_patches",FileStorage::WRITE);
    fs<<"out_patches"<<out_patches;
    fs.release();*/

    return out_patches.t();
}

///used to convert the patches of multi channel images into rows.
///listed in the order of RGB
template<class T>
bool im2col_general(cv::Mat &in_img, vector<cv::Mat> &out_vec, int patch_size, int img_size, int step)
{
    int channels=in_img.channels();

    vector<cv::Mat> layers;
    if(channels>1)
        split(in_img,layers);
    else
        layers.push_back(in_img.t());//to be the same as the demo code, where each patch are extracted in column order

    //in RGB order
    for(int i=channels-1;i>=0;i--)
    {
        cv::Mat patch_mat=im2col_step<T>(layers[i], patch_size, 1);
        out_vec.push_back(patch_mat);
    }

    return true;
}

///show the image from the row vector of CIFAR
void show_img(cv::Mat CIFAR_mat)
{
    //NOTICE: opencv uses BGR when showing images.
    IplImage img;
    /*
    uchar temp_data[IMG_DATA_LENGTH];
    for(int i=0; i<1024; i++)
    {im2col_general
        temp_data[i*3]=CIFAR_mat.data[i+2048];
        temp_data[i*3+1]=CIFAR_mat.data[i+1024];
        temp_data[i*3+2]=CIFAR_mat.data[i];
    }*/
    /*float temp_data[IMG_DATA_LENGTH];
    for(int i=0; i<1024; i++)
    {
        temp_data[i*3]=CIFAR_mat.data[i+2048];
        temp_data[i*3+1]=CIFAR_mat.data[i+1024];
        temp_data[i*3+2]=CIFAR_mat.data[i];
    }
    cv::Mat tmp(IMG_SIZE,IMG_SIZE,CV_32FC3,temp_data);
    img = tmp;*/
    img=CIFAR_mat;
    cvNamedWindow("Image", 1);
    cvShowImage("Image",&img);
    cvWaitKey(0);

    cvDestroyWindow("Image");
    //cvReleaseImage(&&img);
}

void show_multi_img(vector<cv::Mat> img_vec)
{
    //int img_num =
}

int read_CIFAR10(vector<cv::Mat> &train_batch_img, vector<uchar> &train_batch_type, const int stride, bool train)
{
    int sample_toal_num=0;
    int file_num=train? 5:1;

    for(int batch_num=1; batch_num<=file_num; batch_num++)
    {
        char path[100];
        if(train)
            sprintf(path,"/projects/data/cifar-10-batches-bin/data_batch_%d.bin",batch_num);
        else
            //read the test samples
            sprintf(path,"/projects/data/cifar-10-batches-bin/test_batch.bin");

        ifstream batch_file(path,ios::binary);

        //the temporary data
        uchar type;
        uchar tmp[IMG_DATA_LENGTH];
        double *tmpf;
        cv::Mat *im_double;
        //used to subsample
        int sample_num=0;
        while(batch_file.read((char *) &(type),sizeof(char)))
        {

            batch_file.read((char *) tmp,IMG_DATA_LENGTH*sizeof(char));

            if(sample_num++ % stride ==0)
            {
                //convert the scale and data type, and rearrange the order of the pixels
                tmpf= new double[IMG_DATA_LENGTH];
                for(int i=0;i<IMG_DATA_LENGTH/3;i++)
                {
                    tmpf[i*3]=tmp[i+2048];///255.0;
                    tmpf[i*3+1]=tmp[i+1024];///255.0;
                    tmpf[i*3+2]=tmp[i];///255.0;

                }
                //cv::Mat tmp_mat(IMG_SIZE,IMG_SIZE,CV_32FC3,tmpf);
                im_double=new cv::Mat(IMG_SIZE,IMG_SIZE,CV_64FC3,tmpf);

                train_batch_img.push_back(*im_double);
                train_batch_type.push_back(type);


                sample_toal_num++;
            }
        }
        batch_file.close();
    }
        /*FileStorage fs("test_convert1.xml",FileStorage::WRITE);
        fs<<"mat"<<train_batch_img[0];
        fs.release();*/
    //cout<<(int)train_batch_type[70]<<endl;
    //show_img(train_batch_img[70]);


    return sample_toal_num;
}

///convert the pixels from a vector into an image layout
void vec2img(vector<cv::Mat> &CIFAR_data)
{
    uchar temp_data[IMG_DATA_LENGTH];
    int ite_num=IMG_DATA_LENGTH/3;
    int sample_num=CIFAR_data.size();
    for(int k=0;k<sample_num;k++)
    {
        for(int i=0;i<ite_num;i++)
        {
            temp_data[i*3]=CIFAR_data[k].data[i+2048];
            temp_data[i*3+1]=CIFAR_data[k].data[i+1024];
            temp_data[i*3+2]=CIFAR_data[k].data[i];
        }
        cv::Mat tmp(IMG_SIZE,IMG_SIZE,CV_8UC3,temp_data);
        tmp.copyTo(CIFAR_data[k]);
    }
}

///get the PCA filters of each stage
cv::Mat PCA_FilterBank(vector<cv::Mat>& in_img, int patch_size, int num_filters)
{

    //get all the mean-removed patches from the image
    int num_image=in_img.size();
    int size=in_img[0].channels()*patch_size*patch_size;//length of the patch vector
    int in_img_num = in_img.size();
    cv::Mat patch_mean,patch_mean_removed;
    //cv::Mat diff_mat;
    vector<cv::Mat> patch_mat_vec;

    int core_num = omp_get_num_procs();
    //initialize the reduction list
    cv::Mat *Rx_list = new cv::Mat[core_num];
    for(int i=0;i<core_num;i++)
    {
        Rx_list[i] = cv::Mat::zeros(size,size,in_img[0].depth());
    }
    //count time
    cout<<"Number of cores: "<<core_num<<endl;
    struct timeval t1,t2;
    float duration;
    gettimeofday(&t1,NULL);
#pragma omp parallel for default(none) num_threads(core_num) private(patch_mat_vec, patch_mean_removed, patch_mean) shared(in_img,patch_size,in_img_num,size,Rx_list)
    for(int img_num=0; img_num<in_img_num; img_num++)
    {
        //get the patch vector matrix
        im2col_general<double>(in_img[img_num],patch_mat_vec,patch_size,IMG_SIZE,1);
        //remove the mean of patch vectors
        patch_mean_removed.create(0,size,Rx_list[0].type());

        for(vector<cv::Mat>::iterator it2=patch_mat_vec.begin();it2!=patch_mat_vec.end();it2++)
        {
            cv::reduce(*it2,patch_mean,0,CV_REDUCE_AVG);

            for(int i=0;i<it2->rows;i++)
            {
                //must use a intermediate mat
                it2->row(i)=it2->row(i)-patch_mean.row(0);
                patch_mean_removed.push_back(it2->row(i));
            }
        }
        //multiply and add the mean-removed patches
        int k=omp_get_thread_num();
        Rx_list[k] = Rx_list[k] + patch_mean_removed * patch_mean_removed.t();

        //deallocate
        patch_mean_removed.release();
        patch_mat_vec.clear();
    }
    gettimeofday(&t2,NULL);
    //duration = (double)(t2.tv_usec - t1.tv_usec)/1000000;
    duration = t2.tv_sec - t1.tv_sec + (double)(t2.tv_usec - t1.tv_usec)/1000000;
    for(int i=1;i<core_num;i++)
    {
        Rx_list[0] = Rx_list[0]+Rx_list[i];
    }
    cout<<"time usage: "<<duration<<"s"<<endl;

    //used for PCA
    int cols = (IMG_SIZE - patch_size + 1)*(IMG_SIZE - patch_size + 1);
    Rx_list[0]=Rx_list[0] / (double)(num_image*cols);

    //get the eigen value mat and eigen vector mat of Rx
    cv::Mat e_value_mat,e_vec_mat;
    eigen(Rx_list[0],e_value_mat,e_vec_mat);

    //the eigen vectors have probably been sorted by eigen value already
    cv::Mat filters(0,size,Rx_list[0].depth());
    delete[] Rx_list;
    for(int i=0;i<num_filters;i++)
    {
        filters.push_back(e_vec_mat.row(i));
    }

    return filters;
}

///turn the src_img into the result of convolution
///src_label will be cleared and changed!!!
///for stage 0, the out put is the same as demo; for stage 1, the output is the transposition of demo
vector<cv::Mat> PCA_convolution_rmean(vector<cv::Mat> &src_img, cv::Mat &filter_mat, int patch_size)
{
    const cv::Mat ones_mat=cv::Mat::ones(patch_size*patch_size,1,CV_64FC1);
    const cv::Mat filler = cv::Mat::zeros(1,1,CV_64FC1);
    cv::Mat padded_img;//zero-padded image which has one channel
    cv::Mat combined_img;//combined all the channels

    int pad=(patch_size-1)/2;
    cv::Scalar s = cv::Scalar(0);

    const int channels = src_img[0].channels();
    int out_cols = src_img[0].rows*src_img[0].cols;

    cv::Mat product_mat;
    vector<cv::Mat> single_channel_images;//used to store the split images
    vector<cv::Mat> out_mat;
    vector<uchar> out_label;

    const int in_img_num = src_img.size();
    const int num_elements = in_img_num*filter_mat.rows;
    int core_num = omp_get_num_procs();
    int in_img_rows=src_img[0].rows;//suppose all the rows are the same
    int depth = src_img[0].depth();
#pragma omp parallel for default(none) num_threads(core_num) shared(out_mat)
    //allocate the space
    for(int i=0;i<num_elements;i++)
    {
#pragma omp critical
        out_mat.push_back(filler);
    }

#pragma omp parallel for default(none) num_threads(core_num) private(single_channel_images,padded_img,product_mat) shared(depth,src_img,out_cols,pad,s,patch_size,filter_mat,in_img_rows,out_mat)
    for(int img_num=0; img_num<in_img_num; img_num++)
    {
        //split the 3 channels
        if(channels>1)
            split(src_img[img_num],single_channel_images);
        else
            single_channel_images.push_back(src_img[img_num]);

        //turn each channel of image into columns and combine the 3 channels
        cv::Mat temp,patch_mean;
        cv::Mat combiled_img_cols=cv::Mat(0,out_cols,depth);

        for(vector<cv::Mat>::iterator it=single_channel_images.begin();it!=single_channel_images.end();it++)
        {
            cv::copyMakeBorder(*it,padded_img,pad,pad,pad,pad,cv::BORDER_CONSTANT,s);//zero-padding

            temp=im2col_step<double>(padded_img,patch_size,1);//to be the same as the demo
            cv::reduce(temp,patch_mean,0,CV_REDUCE_AVG);
            patch_mean=ones_mat*patch_mean;
            //the mean should be removed
            temp=temp-patch_mean;

            combiled_img_cols.push_back(temp);
        }

        for(int filter_num=0;filter_num<filter_mat.rows;filter_num++)
        {
            product_mat =  filter_mat.row(filter_num)* combiled_img_cols;
            product_mat = product_mat.reshape(0,in_img_rows);
            out_mat[img_num*filter_mat.rows+filter_num] = product_mat;
        }
        single_channel_images.clear();
    }
    src_img.clear();
    return out_mat;
}

///modify the input image
bool Heaviside(cv::Mat &in_img)
{
    int row_num=in_img.rows;
    int col_num=in_img.cols;

    in_img.convertTo(in_img, CV_32FC1);
    float *pt_in_img;

    for(int row=0;row<row_num;row++)
    {
        pt_in_img = in_img.ptr<float>(row);
        for(int col=0;col<col_num;col++)
        {
            pt_in_img[col] = (pt_in_img[col]<0);//to get the same result as the demo
        }
    }
    return true;
}


///each row of block_mat is the vectorized block
cv::Mat Hist(cv::Mat &block_mat, int range)
{
    int row_num=block_mat.rows;//the range of histogram is from 0~range
    cv::Mat BHist = cv::Mat::zeros(row_num,range+1,block_mat.depth());

    float *pt_in,*pt_out;

    for(int row=0;row<row_num;row++)
    {
        pt_in = block_mat.ptr<float>(row);
        pt_out = BHist.ptr<float>(row);
        for(int col=0;col<block_mat.cols;col++)
        {
            pt_out[(int)pt_in[col]]++;
        }
    }

    /*FileStorage fs("test_BHist.xml",FileStorage::WRITE);
    fs<<"BHist"<<BHist;
    fs.release();*/

    BHist=BHist.t();

    return BHist;
}

///again assume that the input images are square
cv::Mat Spp(cv::Mat &BHist, const PCANet PCANet_params, const int img_width)
{
    int start_idx = PCANet_params.histBlockSize[0]/2;
    int end_idx = img_width-PCANet_params.histBlockSize[0]/2;
    int stride = (1-PCANet_params.blkOverlapRatio)*PCANet_params.histBlockSize[0];
    int layers = PCANet_params.pyramid.size();
    //get the dimension of features
    int feature_num = 0;
    for(int i=0;i<layers;i++)
    {
        feature_num += PCANet_params.pyramid[i]*PCANet_params.pyramid[i];
    }
    cv::Mat out_mat = cv::Mat::zeros(BHist.rows,feature_num,BHist.depth());

    //map the features to the corresponding indexes in out_mat
    int idx;
    int offset=0;
    for(int layer=0;layer<layers;layer++)
    {
        int cnt=0;
        int step = img_width/PCANet_params.pyramid[layer];
        for(int row=start_idx; row<=end_idx; row+=stride)
        {
            for(int col=start_idx; col<=end_idx; col+=stride)
            {

                idx = (row/step)*PCANet_params.pyramid[layer]+col/step+offset;
                out_mat.col(idx) = cv::max(BHist.col(cnt),out_mat.col(idx));
                cnt++;
            }
        }
        //plus this offset!
        offset = offset+pow(2,PCANet_params.pyramid[layer]);
    }

    return out_mat;
}

///return value is a row vector of all the histograms
///output mat is of dimension 10240*21
cv::Mat HashingHist(vector<cv::Mat> &in_img, const PCANet PCANet_params)
{
    int num_in_img=in_img.size();
    int num_filters=PCANet_params.numFilters[PCANet_params.numStages-1];
    int num_img0 = num_in_img/num_filters;
    int row_in_img = in_img[0].rows;
    int col_in_img = in_img[0].cols;

    //get the weight constant
    float *w = new float[num_filters];
    w[num_filters-1]=1;
    for(int i=num_filters-2;i>=0;i--)
    {
        w[i]=w[i+1]*2;
    }
    int range = pow(2,num_filters)-1;

    //get the stride constant, assume that the blocks are squares
    int stride = (1-PCANet_params.blkOverlapRatio)*PCANet_params.histBlockSize[0];

    vector<cv::Mat>::iterator it_in_img = in_img.begin();
    cv::Mat BHist;//used to store the histogram
    for(int i=0;i<num_img0;i++)
    {
        cv::Mat T = cv::Mat::zeros(row_in_img,col_in_img,CV_32FC1);//use higher accuracy to prevent overflow

        for(int j=0;j<num_filters;j++)
        {
            Heaviside(*it_in_img);

            T = T + w[j]* (*it_in_img);
            //it_in_img = in_img.erase(it_in_img);
            it_in_img->release();
            ++it_in_img;
        }

        cv::Mat blocks_mat = im2col_step<float>(T,PCANet_params.histBlockSize[0],stride);

        blocks_mat=blocks_mat.t();
        blocks_mat = Hist(blocks_mat,range);

        if(PCANet_params.pyramid.size()!=0)
            blocks_mat = Spp(blocks_mat,PCANet_params,row_in_img);

        if(i==0)
            BHist = blocks_mat;
        else
            vconcat(BHist,blocks_mat,BHist);
    }
    delete[] w;

    //use spatial pyramid pooling when the parameters are set, and vectorize
    //BHist = BHist.reshape(0,1);

    return BHist;
}


///get the patterns of PCA filters
///the label will match the last stage only
PCATrainResult *PCANet_train(vector<cv::Mat> &train_batch_img, vector<uchar> &train_batch_type, const PCANet PCANet_params, bool extract_feature)
{
    cv::Mat filler = cv::Mat::zeros(1,1,CV_32FC1);
    PCATrainResult *train_result;
    int num_inImg=train_batch_img.size();
    vector<cv::Mat> out_image = train_batch_img;
    train_batch_img.clear();
    vector<uchar> out_type = train_batch_type;
    //allocate storage
    train_result=new PCATrainResult;

    //extract the PCA filters
    for(int stage=0;stage<PCANet_params.numStages;stage++)
    {
        //get the filters
        cout<<"Getting PCA filters of stage "<<stage<<"...."<<endl;
        train_result->filters.push_back(PCA_FilterBank(out_image,PATCH_SIZE,PCANet_params.numFilters[stage]));
        //out_mat("filter1",train_result->filters[0]);
        //do convolution for each intermediate layer
        cout<<"Doing convolution of stage 0..."<<endl;
        if(stage != PCANet_params.numStages-1)
            out_image=PCA_convolution_rmean(out_image,train_result->filters[stage],PCANet_params.patchSize[stage]);
    }
    //out_image.clear();
//should be 15G here if use the full dataset
    cout<<"Number of images after the first convolutions: "<<out_image.size()<<", size: "<<out_image[0].cols<<endl;

    struct timeval t1,t2;
    cout<<"Extracting hist features of the training samples..."<<endl;
    gettimeofday(&t1,NULL);
    if(extract_feature)
    {
        vector<cv::Mat>::iterator out_img_begin=out_image.begin();

        //initialize hashing_result for parallelism
        int core_num = omp_get_num_procs();
        ///large memory space
        vector<cv::Mat> hashing_result_vec;
        for(int i=0;i<num_inImg;i++)
        {
            hashing_result_vec.push_back(filler);
        }
        //to reduce memory usage, firstly get all the features, then perform PCA for all the 21 blocks
        // if parallelism is used, out_image won't be able to free...
//#pragma omp parallel for default(none) num_threads(core_num) shared(num_inImg,train_result,hashing_result_vec,out_img_begin)
        for(int i=0;i<num_inImg;++i)
        {
            vector<cv::Mat> sub_inImg(out_img_begin+i*PCANet_params.numFilters[0],out_img_begin+(i+1)*PCANet_params.numFilters[0]);

            sub_inImg = PCA_convolution_rmean(sub_inImg,train_result->filters[1],PCANet_params.patchSize[1]);

            cv::Mat hashing_result = HashingHist(sub_inImg,PCANet_params);

            hashing_result_vec[i]=hashing_result.t();

        }//55G here
        out_image.clear();//effective only when the above is not parallel
        gettimeofday(&t2,NULL);
        double duration = t2.tv_sec-t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
        cout<<"time usage: "<<duration<<"s"<<endl;

        cout<<"getting PCA vectors..."<<endl;
        gettimeofday(&t1,NULL);
        for(int i=0;i<PCANet_params.block_num;i++)
        {
            cv::Mat tmp;
            train_result->e_vec.push_back(tmp);
        }
        //create a stack to store the reduced vectors in order to save memory
        //each element contain the 21 reduced vectors of each image
        cv::Mat *hashing_result_stacks = new cv::Mat[num_inImg];
        cout<<"start!"<<endl;
#pragma omp parallel for default(none) num_threads(core_num) shared(num_inImg, hashing_result_stacks)
        for(int i=0;i<num_inImg;++i)
        {
            hashing_result_stacks[i].create(0,PCANet_params.dim_features,CV_32FC1);
        }
#pragma omp parallel for default(none) num_threads(4) shared(core_num,num_inImg,hashing_result_vec,hashing_result_stacks,train_result)
        for(int blk_num=PCANet_params.block_num-1;blk_num>=0;--blk_num)
        {
            struct timeval t3;
            gettimeofday(&t3,NULL);
            //at most 1.9G. used to extract the feature vectors. all the feature vectors of one bin
            cv::Mat vecs_of_block = cv::Mat::zeros(num_inImg,10240,CV_32FC1);
//#pragma omp parallel for default(none) num_threads(core_num) shared(vecs_of_block,num_inImg,hashing_result_vec,blk_num)
            for(int img_num=0;img_num<num_inImg;++img_num)
            {
                cv::Mat M = hashing_result_vec[img_num].row(blk_num);
                M.copyTo(vecs_of_block.row(img_num));
            }

            vecs_of_block=vecs_of_block.t();

            ///test for Eigen
            cv::Mat eigen_vec;
            MatrixXf eigen_mat;
            cv2eigen(vecs_of_block,eigen_mat);
            JacobiSVD<MatrixXf> svd(eigen_mat,ComputeThinU);
            eigen_mat.resize(0,0);
            eigen2cv(svd.matrixU(),eigen_vec);

            /*cv::Mat eigen_vec;
            MatrixXf eigen_mat;
            cv2eigen(vecs_of_block,eigen_mat);
            eigen_mat = eigen_mat * eigen_mat.transpose();
            EigenSolver<MatrixXf> e_solver(eigen_mat,true);
            eigen2cv(e_solver.eigenvectors(),eigen_vec);*/

            //out_mat<float>("test_eigen",eigen_vec);
            eigen_vec = eigen_vec.t();
            /*out_mat<float>("A",vecs_of_block);
            out_mat<float>("W",singular_value);
            out_mat<float>("U",eigen_vec);
            out_mat<float>("Vt",Vt);*/
            //here rows should be the same as the sample num... empirically
            if(eigen_vec.rows > 1280)
            {
                eigen_vec.pop_back(eigen_vec.rows-1280);//now it has 1280 rows
            }
            //out_mat<float>("eigen_vec",eigen_vec);
//#pragma omp parallel for default(none) num_threads(core_num) shared(hashing_result_stacks,hashing_result_vec,num_inImg,eigen_vec,blk_num)
            for(int img_num=0;img_num<num_inImg;++img_num)
            {
                 cv::Mat reduced_vec = hashing_result_vec[img_num].row(blk_num) * eigen_vec.t();
                 hashing_result_stacks[img_num].push_back(reduced_vec);
                 hashing_result_vec[img_num].pop_back(1);//in exchange for memory space
            }

            train_result->e_vec[blk_num]=eigen_vec;//need to revise the order when testing

            struct timeval t4;
            gettimeofday(&t4,NULL);
            float dura = t4.tv_sec-t3.tv_sec + (t4.tv_usec - t3.tv_usec)/1000000.0;
            char filename[100];
            sprintf(filename,"%d_%f",blk_num,dura);
            ofstream of(filename);
            of.close();
            //cout<<"Finished PCA for block "<<blk_num<<". "<<"Time usage: "<<duration<<"s"<<endl;
        }
        gettimeofday(&t2,NULL);
        duration = t2.tv_sec-t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
        cout<<"time usage: "<<duration<<"s"<<endl;

        cout<<"copying to train_result..."<<endl;
        gettimeofday(&t1,NULL);
        train_result->feature = cv::Mat::zeros(num_inImg,PCANet_params.dim_features*21,CV_32FC1);
#pragma omp parallel for default(none) num_threads(core_num) shared(num_inImg,hashing_result_stacks,train_result)
        for(int img_num = 0;img_num<num_inImg;img_num++)
        {
            hashing_result_stacks[img_num] = hashing_result_stacks[img_num].reshape(0,1);
            hashing_result_stacks[img_num].copyTo(train_result->feature.row(img_num));
            hashing_result_stacks[img_num].release();
        }

        delete []hashing_result_stacks;
        gettimeofday(&t2,NULL);
        duration = t2.tv_sec-t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
        cout<<"time usage: "<<duration<<"s"<<endl;
    }

    //out_file("stage1",train_result->filters[0]);
    //out_file("stage2",train_result->filters[1]);

    //save the result
    //set the label
    float *labels = new float[train_batch_type.size()];
    for(int i=0;i<train_batch_type.size();i++)
        labels[i] = train_batch_type[i];

    cv::Mat temp(train_batch_type.size(),1,CV_32FC1,labels);
    train_result->label=temp;
    out_liblinear("train_liblinear",train_result->feature,temp);
    train_batch_type.clear();

    return train_result;
}

///SVM parameters are set here
void SVMTrain(cv::Mat &features, cv::Mat &labels, CvSVM &SVM)
{
    cout <<"\n ====== Training Linear SVM Classifier ======= \n"<<endl;
    //set the parameters
    CvSVMParams SVM_params;
    SVM_params.svm_type = CvSVM::C_SVC;
    SVM_params.C = 10;
    SVM_params.kernel_type = CvSVM::LINEAR;
    SVM_params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);//终止准则函数：当迭代次数达到最大值时终止
    //out_mat("features",features);
    //out_mat("labels",labels);

    //create an instance of CvSVM
    int64 e1=cv::getTickCount();
    SVM.train(features,labels,Mat(),Mat(),SVM_params);
    features.release();
    labels.release();
    int64 e2=cv::getTickCount();
    float train_time = (e2-e1)/cv::getTickFrequency();
    //delete them


    cout<<"svm training complete, time usage: "<<train_time<<endl;

}
///assume there are 2 stages only
double test_accuracy(vector<cv::Mat> &test_batch_img, vector<uchar> &test_batch_type,
                     vector<cv::Mat> &filters, vector<cv::Mat> PCA_vecs, CvSVM &SVM, PCANet &PCANet_params)
{
    int correct_num[8]={0};
    int total_num=test_batch_img.size();

    vector<cv::Mat>::iterator ite_img =test_batch_img.begin();
    vector<uchar>::iterator ite_label = test_batch_type.begin();
    cv::Mat hashing_result_store = cv::Mat::zeros(total_num,PCANet_params.dim_features*PCANet_params.block_num,CV_32FC1);
    int core_num = omp_get_num_procs();
#pragma omp parallel for default(none) num_threads(core_num) shared(filters, PCA_vecs, total_num, PCANet_params, hashing_result_store, test_batch_img, SVM,test_batch_type,correct_num)
    for(int i=0;i<total_num;i++)
    {
        vector<cv::Mat> temp;
        temp.push_back(test_batch_img[i]);

        temp=PCA_convolution_rmean(temp,filters[0],PCANet_params.patchSize[0]);

        temp=PCA_convolution_rmean(temp,filters[1],PCANet_params.patchSize[0]);

        cv::Mat hashing_result = HashingHist(temp,PCANet_params);
        cv::Mat hashing_result_t(0,PCANet_params.dim_features,CV_32FC1);
        for(int blk_num=PCANet_params.block_num-1;blk_num>= 0 ; --blk_num)
        {
            cv::Mat tmp = PCA_vecs[blk_num] * hashing_result.col(blk_num);
            tmp = tmp.t();
            hashing_result_t.push_back(tmp);
        }
        hashing_result = hashing_result_t.reshape(0,1);
        hashing_result.copyTo(hashing_result_store.row(i));

        test_batch_img[i].release();

        float prediction = SVM.predict(hashing_result);
        int k=omp_get_thread_num();
        if((uchar)prediction == test_batch_type[i])
            correct_num[k]++;
    }

    out_liblinear2("test_set",hashing_result_store,test_batch_type);
    //reduce
    for(int i=1;i<core_num;i++)
    {
        correct_num[0] += correct_num[i];
    }

    double accuracy = (double)correct_num[0]/total_num;
    return accuracy;
}

int main(int argc, char **argv)
{

    const int SAMPLE_STRIDE=4;//used for subsampling
    vector<cv::Mat> train_batch_img;
    vector<uchar> train_batch_type;
    vector<cv::Mat> test_batch_img;
    vector<uchar> test_batch_type;
    PCATrainResult *PCA_train_result;
    CvSVM SVM;

    int sample_toal_num;//the total number of samples

    ///set the parameters of PCAnet
    PCANet PCANet_params;
    PCANet_params.numStages=2;
    PCANet_params.numFilters.push_back(40);
    PCANet_params.numFilters.push_back(8);
    PCANet_params.patchSize.push_back(5);
    PCANet_params.patchSize.push_back(5);
    PCANet_params.histBlockSize.push_back(8);
    PCANet_params.histBlockSize.push_back(8);
    PCANet_params.blkOverlapRatio=0.5;
    PCANet_params.pyramid.push_back(4);
    PCANet_params.pyramid.push_back(2);
    PCANet_params.pyramid.push_back(1);
    PCANet_params.block_num = 21;
    PCANet_params.dim_features = (50000/SAMPLE_STRIDE > 1280) ? 1280 : (50000/SAMPLE_STRIDE);


    ///read CIFAR-10 training data
    sample_toal_num=read_CIFAR10(train_batch_img,train_batch_type,SAMPLE_STRIDE,true);
    //show_img(train_batch[0].img);
    cout<<"Use "<<sample_toal_num<<" samples for training."<<endl;

    //PCA_FilterBank(train_batch_img,5,1);//test this function

    PCA_train_result = PCANet_train(train_batch_img,train_batch_type,PCANet_params,true);


    SVMTrain(PCA_train_result->feature,PCA_train_result->label,SVM);


    sample_toal_num=read_CIFAR10(test_batch_img,test_batch_type,10,false);
    //show_img(train_batch[0].img);
    cout<<"Use "<<sample_toal_num<<" samples for testing."<<endl;

    double accuracy = test_accuracy(test_batch_img,test_batch_type,PCA_train_result->filters,PCA_train_result->e_vec, SVM,PCANet_params);

    cout<<accuracy<<endl;

    delete PCA_train_result;

    getchar();

    return 0;
}


