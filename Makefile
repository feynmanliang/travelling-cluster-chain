include /usr/local/conf/ElVars

CC=g++
MPICC=mpic++

CFLAGS=$(EL_COMPILE_FLAGS) -O3 -Wall -Wno-sign-compare --std=c++11
LDFLAGS=-lgsl
OBJ_PATH = ./obj
BIN_PATH = ./bin


all: mpi_gaussian mpi_gaussian_imbalance mpi_lda_sgrld mpi_lda_testdata


clean:
	rm -rf $(OBJ_PATH)
	rm -rf $(BIN_PATH)
	mkdir -p $(BIN_PATH)

# OBJ_SRCS := cmd_flags.cc common.cc document.cc model.cc accumulative_model.cc sampler.cc
OBJ_SRCS := sampler.cc sgld_sampler.cc gmm_toy_model.cc sgrld_sampler.cc lda_model.cc
ALL_OBJ = $(patsubst %.cc, %.o, $(OBJ_SRCS))
OBJ = $(addprefix $(OBJ_PATH)/, $(ALL_OBJ))

$(OBJ_PATH)/%.o: %.cc
	@ mkdir -p $(OBJ_PATH)
	$(CC) -c $(CFLAGS) $(LDFLAGS) $< -o $@

lda: lda.cc $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJ) $< -o $@

infer: infer.cc $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJ) $< -o $@

mpi_lda: mpi_lda.cc $(OBJ)
	$(MPICC) $(CFLAGS) $(LDFLAGS) $(OBJ) $< -o $@

mpi_gaussian: mpi_gaussian.cc $(OBJ)
	$(MPICC) $(CFLAGS) $(LDFLAGS) $(OBJ) $< -o ${BIN_PATH}/$@ $(EL_LINK_FLAGS) $(EL_LIBS)

mpi_gaussian_imbalance: mpi_gaussian_imbalance.cc $(OBJ)
	$(MPICC) $(CFLAGS) $(LDFLAGS) $(OBJ) $< -o ${BIN_PATH}/$@ $(EL_LINK_FLAGS) $(EL_LIBS)

mpi_lda_sgrld: mpi_lda_sgrld.cc $(OBJ)
	$(MPICC) $(CFLAGS) $(LDFLAGS) $(OBJ) $< -o ${BIN_PATH}/$@ $(EL_LINK_FLAGS) $(EL_LIBS)

mpi_lda_testdata: mpi_lda_testdata.cc $(OBJ)
	$(MPICC) $(CFLAGS) $(LDFLAGS) $(OBJ) $< -o ${BIN_PATH}/$@ $(EL_LINK_FLAGS) $(EL_LIBS)
