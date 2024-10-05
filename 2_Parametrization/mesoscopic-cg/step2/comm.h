#ifndef COMM_H
#define COMM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VAR_BEGIN { \
  FILE* varfile = fopen("config.in", "r"); \
  char line[1024]; \
  while(fgets(line, 1023, varfile)) { char* p = strchr(line,'='); if(!p) continue; \
  char *pn = strchr(p+1, '\n'); if(pn) pn[0] = 0; bool ff=true; \
  p[0]=0; p++; if(false);

#define VAR_END else {ff=false;} if(ff) printf("%s = %s\n", line, p);} fclose(varfile); printf("\n"); }

#define GET_INT(key)	else if(strcmp(line,#key)==0) { key = atoi(p); }
#define GET_REAL(key)	else if(strcmp(line,#key)==0) { key = atof(p); }
#define GET_STRING(key)	else if(strcmp(line,#key)==0) { strcpy(key,p); }

#endif
