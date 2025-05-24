



class DoNothing:
    def forward(self, vit_embeds, *args, **kwargs):
        return vit_embeds