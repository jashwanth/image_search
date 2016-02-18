#include "files.h"
#define no_of_clusters 7500
int thresh = 200;
int max_thresh = 255;
using namespace std;
using namespace cv;
Mat img;
string source_window("Source image");
string corners_window("Corners detected");
int main(int argc,char *argv[])
{
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
	Ptr<FeatureDetector> detector = FeatureDetector::create("DynamicSIFT");
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
	BOWImgDescriptorExtractor bowDE(extractor,matcher);
	Mat descriptors;
	Mat training_descriptors(1,extractor->descriptorSize(),extractor->descriptorType());
	Mat img;
	SiftFeatureDetector sif;
	SiftDescriptorExtractor sif_ex;
	string dir = "./Caltech_11classes",filepath;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;
	vector<KeyPoint> keypoints;
	dp = opendir( dir.c_str() );
	cout << "------- build vocabulary ---------\n";
	cout << "extract descriptors.."<<endl;
	while ((dirp = readdir( dp )))
	{
		if (strcmp(dirp->d_name,".")==0) continue;
		if (strcmp(dirp->d_name,"..")==0) continue;
		filepath = dir + "/" + dirp->d_name;
//		cout<<"HI"<<endl;
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
//				cout<<temp_filename<<endl;
				if (stat(temp_filename.c_str(),&temp_filestat)) continue;
				img = imread(temp_filename,0);
				if (!img.data) continue;
				sif.detect(img, keypoints);
				extractor->compute(img, keypoints, descriptors);
				training_descriptors.push_back(descriptors);
//				cout << ".";
				count++;
			}
			closedir(temp_dp);
		}
	}
	closedir(dp);
	cout << "Total descriptors: " << training_descriptors.rows << endl;
	FileStorage fs("training_descriptors_sift.yml", FileStorage::WRITE);
	fs << "training_descriptors_sift" << training_descriptors;
	fs.release();
	BOWKMeansTrainer bowtrainer(no_of_clusters); //num clusters
	bowtrainer.add(training_descriptors);
	cout << "cluster BOW features" << endl;
	Mat vocabulary = bowtrainer.cluster();
	FileStorage fs1("vocabulary_sift.yml", FileStorage::WRITE);
	fs1 << "vocabulary_sift" << vocabulary;
	fs1.release();
	return 0;
}
