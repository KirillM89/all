#include <gtest/gtest.h>
int main(int nargs, char** argv)
{
    testing::InitGoogleTest(&nargs, argv);
    return RUN_ALL_TESTS();
}