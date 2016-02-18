#include "files.h"
int main(int argc,char** argv)
{
  Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
  Ptr<DescriptorMatcher> matcher     = DescriptorMatcher::create("FlannBased");
  SurfFeatureDetector sif(400);
  BOWImgDescriptorExtractor bowDE(extractor,matcher);
  Mat bowDescriptor;
  FileStorage fs("vocabulary_surf.yml", FileStorage::READ);
  Mat vocal,img;
  fs["vocabulary_surf"] >> vocal;
  fs.release();
  bowDE.setVocabulary(vocal);
//  vector<set<string> > inv_index(vocal.rows);
  map<string,int> file_index;
  Mat inv_index = Mat::zeros(vocal.rows,330,CV_32FC1);
  Mat doc_vec = Mat::zeros(vocal.rows,vocal.cols,CV_32FC1);
  string dir = "./Caltech_11classes",filepath;
  DIR *dp;
  struct dirent *dirp;
  struct stat filestat;
  dp = opendir( dir.c_str());
  int j=0;
  while ((dirp = readdir(dp)))
  {
    if (strcmp(dirp->d_name,".")==0) continue;
    if (strcmp(dirp->d_name,"..")==0) continue;
    filepath = dir + "/" + dirp->d_name;
    cout<<filepath<<endl;
    if (stat( filepath.c_str(),&filestat)) continue;
    if (S_ISDIR(filestat.st_mode)){
      DIR *temp_dp;
      struct dirent *temp_dirp;
      struct stat temp_filestat;
      string temp_filename;
      temp_dp=opendir(filepath.c_str());
      int count=0;
      while ((temp_dirp=readdir(temp_dp)) and (count<30)){
	if (strcmp(temp_dirp->d_name,".")==0) continue;
	if (strcmp(temp_dirp->d_name,"..")==0) continue;
	temp_filename=filepath+"/"+temp_dirp->d_name;
	if (stat(temp_filename.c_str(),&temp_filestat)) continue;
	img = imread(temp_filename,0);
	if (!img.data) continue;
	vector<KeyPoint> keys;
	vector<vector<int> > v;
	Mat key_desc;
	sif.detect(img,keys);
	bowDE.compute(img,keys,bowDescriptor,&v,&key_desc);
	for (int i=0;i<(int)v.size();i++){
	  if (v[i].size()>0){
	    inv_index.at<float>(i,j)=v[i].size();
	  }
	}
	j++;
	count++;
      }
      closedir(temp_dp);
    }
  }
  FileStorage fs1("inverted_index_surf.yml", FileStorage::WRITE);
  fs1<<"inverted_index_surf"<<inv_index;
/*  for (int i=0;i<5000;i++){
    int nd=0,n=0;
    for (j=0;j<40;j++){
      nd+=inv_index.at<float>(i,j);   // nd = total occurances of ith vocabulary word
      if ((inv_index.at<float>(i,j))>0) n++;  // n = total no . of documents which contain ith vocabulary
    }
    for (j=0;j<40;j++){
      if (n!=0) doc_vec.at<float>(i,j)=(inv_index.at<float>(i,j)/nd)*(40/n);
      else doc_vec.at<float>(i,j)=0;
    }
  }
  fs1<<"document vector"<<doc_vec;*/
  fs1.release();
}
