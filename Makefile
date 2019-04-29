SUBDIRS = pytorch/ATan pytorch/autograd pytorch/python tensorflow/basic tensorflow/python cudnn/cudnnver
SUBDIRSCLEAN = $(addsuffix .clean,$(SUBDIRS))

.PHONY: all subdirs $(SUBDIRS) $(SUBDIRSCLEAN) message clean

all: message

subdirs: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ run

analysis: common

message: subdirs
	@echo '#########################################'
	@echo '############### COMPLETED ###############'
	@echo '#########################################'

clean: $(SUBDIRSCLEAN)

$(SUBDIRSCLEAN): %.clean:
	$(MAKE) -C $* clean
