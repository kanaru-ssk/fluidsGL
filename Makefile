CC=nvcc
CFLRAGS=

TARGETDIR=bin
TARGET=$(TARGETDIR)/app

SRCDIR=src
SRCS=$(shell find $(SRCDIR) -name '*.cu' -o -name '*.cpp')

OBJDIR=obj
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .cpp,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))

INCDIR = -I../../common/inc

LIBDIR = -L../../common/lib/linux/aarch64
LIBS = -lGL -lglut -lGLU -lcufft

$(TARGET): $(OBJS)
	[ -d $(TARGETDIR) ] || mkdir $(TARGETDIR)
	$(CC) $(CFLRAGS) $+ -o $@ $(LIBDIR) $(LIBS)

$(SRCDIR)/%.cpp: $(SRCDIR)/%.cu
	$(CC) $(CFLRAGS) $(INCDIR) --cuda $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(CC) $(CFLRAGS) $(INCDIR) $< -c -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(TARGET)