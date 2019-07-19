#ifndef IMAGEMANAGER_H
#define IMAGEMANAGER_H

#include <stdio.h>
#include <iostream>
#include <vector>
#include <dirent.h>
#include <algorithm>
#include <string.h>

class ImageManager
{
    public:
        ImageManager(const std::string &d);
        ~ImageManager();
        inline int getCount() { return count; }
        inline int getEnd() { return end; }
        std::string next(int step);
        std::string current(int c);
        void sorting(std::vector<std::string>& data);
        int count, end;
        std::vector<std::string> filename;
};

#endif // IMAGEMANAGER_H
