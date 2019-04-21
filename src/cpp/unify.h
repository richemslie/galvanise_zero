
void setDirichletNoise(PuctNode* node);
float priorScore(PuctNode* node, int depth) const;
void setPuctConstant(PuctNode* node, int depth) const;
float getTemperature(int depth) const;

const PuctNodeChild* choose(const PuctNode* node);
bool converged(int count) const;
