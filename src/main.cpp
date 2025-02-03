
#include "cpp_TP1.hpp"
#include "cuda_TP1.hpp"

#include "cpp_TP2.hpp"
#include "cuda_TP2.hpp"

#include "cpp_TP3_glob.hpp"
#include "cuda_TP3_glob.hpp"

#include "cpp_TP3_shrd.hpp"
#include "cuda_TP3_shrd.hpp"

int main(int, char*[])
{
	//nx2_plus_ny_GPU();
	//check_prime_glob_GPU();
	check_prime_shrd_GPU();
	return 0;
}