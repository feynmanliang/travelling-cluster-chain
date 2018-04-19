include /usr/local/conf/ElVars

CC=g++
MPICC=mpic++

CFLAGS=$(EL_COMPILE_FLAGS) -O3 -Wall -Wno-sign-compare --std=c++11
OBJ_PATH = ./obj

all: mpi_gaussian mpi_gaussian_imbalance

clean:
	rm -rf $(OBJ_PATH)
	rm -f lda mpi_lda infer

# OBJ_SRCS := cmd_flags.cc common.cc document.cc model.cc accumulative_model.cc sampler.cc
OBJ_SRCS := sampler.cc
ALL_OBJ = $(patsubst %.cc, %.o, $(OBJ_SRCS))
OBJ = $(addprefix $(OBJ_PATH)/, $(ALL_OBJ))

$(OBJ_PATH)/%.o: %.cc
	@ mkdir -p $(OBJ_PATH)
	$(CC) -c $(CFLAGS) $< -o $@

lda: lda.cc $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $< -o $@

infer: infer.cc $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $< -o $@

mpi_lda: mpi_lda.cc $(OBJ)
	$(MPICC) $(CFLAGS) $(OBJ) $< -o $@

mpi_gaussian: mpi_gaussian.cc $(OBJ)
	$(MPICC) $(CFLAGS) $(OBJ) $< -o $@ $(EL_LINK_FLAGS) $(EL_LIBS)

mpi_gaussian_imbalance: mpi_gaussian_imbalance.cc $(OBJ)
	$(MPICC) $(CFLAGS) $(OBJ) $< -o $@ $(EL_LINK_FLAGS) $(EL_LIBS)

mpi_lda_mcmc: mpi_lda_mcmc.cc $(OBJ)
	$(MPICC) $(CFLAGS) $(OBJ) $< -o $@
