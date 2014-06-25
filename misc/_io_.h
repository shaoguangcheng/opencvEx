/**
 *  @file io.h
 *  @brief In this file, I plan to rewrite some useful but simple I/O functions
 *
 *  @brief If you have any questions about this code,please contact me directly.
 *  @brief Email : chengshaoguang@gmail.com
 *
 *  @brief Write by Shaoguang Cheng at Northwest Polytechnical University
 */

#ifndef _IO_H_
#define _IO_H_

#include <iostream>
#include <cassert>

namespace csg{
/// define print class to output expressions like python
    class print{
    public:
        print() : space(false){}
        ~print() {
            std::cout << std::endl; // print enter after each recall
        }

        /// overload operator ','
        template <class T>
        print& operator , (const T &p) {
            if(space)
                std::cout << " ";
            else
                space = true;
            std::cout << p;

            return *this;
        }
    private:
        bool space;
    };

    /// output 1d array
    /**
     * This function is used to output one dimision array
     * @param p  the name of one dimidion array
     * @param length  the length of one dimision array
     * @param separator
     */
    template <class T>
    void print1d(T *p,int length , char *separator = ""){
        assert(p != NULL&&length >= 0);
        T *t = p;
        while(length--)
            std::cout << *(t++) << separator;
        std::cout << std::endl;
     }

    /// output 2d array
    /**
     *  This function is used to output two dimision array
     *  @param p  the start address of the two dimision array. For example, if the two dimision array is a[M][N], then parameter p must be a[0] rather than a
     *  @param len1d  the first dimision length of the two dimision array
     *  @param len2d  the second dimision length of the two dimision array
     *  @param separator
      */
    template <class T>
    void print2d(T *p,int len1d,int len2d,char *separator = ""){
        assert(p != NULL&&len1d >=0 &&len2d >= 0);
        T *t = p;
        int len = len1d*len2d;
        while(len--)
            std::cout << *(t++) << separator;
        std::cout << std::endl;
    }

    /// output container
    /**
     * This function is used to output container type data.
     * @param container  the container data you want to output. For example, vector, list, deque, set and so on.
     * @param separator
     */
    template <class T>
    void printCon(const T &container, char *separator = "")
    {
        typename T::const_iterator iterator;
        for(iterator = container.begin();iterator != container.end();iterator++)
            std::cout << *iterator << separator;
        std::cout << std::endl;
    }
}

#define PRINT csg::print(),

/// test print class
void testPrint()
{
    int a = 1,b = 2;
    PRINT "this is a test";
    PRINT "the sum of" , a , "and" , b , "is" , a+b;
}

/// test print1d function
#include <string.h>
void testPrint1d()
{
    const char s[] = "Linux code\n";
    csg::print1d(s,strlen(s));
}

/// test print2d function
void testPrint2d()
{
    int a[2][3] = {{1,2,3},{5,6,7}};
    csg::print2d(a[0],3,2," ");
}

/// test print container function
#include <list>
void testPrintCon()
{
    std::list<int> l;
    for(int i=0;i<10;i++)
        l.push_front(i);

    csg::printCon(l," ");
}
#endif
