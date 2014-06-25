#ifndef DIRECTORY_H
#define DIRECTORY_H

#include <vector>
#include <string>
#include <cstring>
#include <iostream>

#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

using namespace std;

namespace csg {
    typedef vector<string> strVec;

/**
     * @brief isDir test if dir is a directory
     * @param dir
     * @return
     */
    bool isDir(const char* dir)
    {
        if(dir == NULL)
            return false;

        struct stat statBuf;
        lstat(dir,&statBuf);
        int ret = S_ISDIR(statBuf.st_mode);
        if(ret == 1) return true;

        return false;
    }


    /// return the current path
    /// @param path  return value
    int pwd(string& path)
    {
        char buffer[1024];
        if(getcwd(buffer,sizeof(buffer)) == NULL){
            cout << "directory name is too long\n";
            return -1;
        }

        path = string(buffer);
        return 0;
    }

    /// return the current directory name
    int currentDirectoryName(string& directoryName)
    {
        string path;

        pwd(path);
        basic_string<char>::size_type pos = path.find_last_of("/");

        directoryName.resize(path.size()-pos);
        copy(path.begin()+pos+1,path.end(),directoryName.begin());

        return 0;
    }

    /// enter a directory
    void cd(const string& path)
    {
        chdir(path.c_str());
    }

    /// return the files name in specified directory,recursively
    int getFiles(const string path,strVec& fileName)
    {
        DIR    *dir = NULL;
        struct dirent *entry = NULL;
        struct stat   stateBuf;

        if((dir = opendir(path.c_str())) == NULL){
            cout << "can not open the directory\n";
            return -1;
        }

        cd(path);

        while((entry = readdir(dir)) != NULL){
            lstat(entry->d_name,&stateBuf);
            if(S_ISDIR(stateBuf.st_mode)){
                if(strcmp(".",entry->d_name) == 0||
                        strcmp("..",entry->d_name) == 0||
                        entry->d_name[0] == '.')         //ignore the hidden folder
                    continue;

                getFiles(string(entry->d_name),fileName);
            }

			lstat(entry->d_name,&stateBuf);
            if(entry->d_name[0] != '.'&&!S_ISDIR(stateBuf.st_mode)){
				string p="";
				pwd(p);
                fileName.push_back(p+'/'+string(entry->d_name));
			}
        }

        cd("..");
        closedir(dir);

        return 0;
    }

    bool isDirectoryExsit(const char* path)
    {
        if(NULL == path)
            return false;

        DIR *dir = NULL;
        dir = opendir(path);
        if( NULL == dir){
            printf("%s does not exsit\n",path);
            return false;
        }

        return true;
    }

    bool isFileExsit(const char* file)
    {
        if(NULL == file)
            return false;
        if(access(file,F_OK) != 0){
            printf("%s does not exsit\n",file);
            return false;
        }

        return true;
    }

}//end of namespace csg

#endif // DIRECTORY_H
