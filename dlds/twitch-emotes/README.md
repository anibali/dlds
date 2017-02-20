# Twitch emotes

https://twitchemotes.com/sub/1

## Structure

DLDS will install the following files for this dataset:

```
twitch-emotes
├── classes.txt
└── twitch-emotes.h5
```

`twitch-emotes.h5` contains the images and labels for the training and test
sets. Its internal structure is as follows:

| Name                          | Type      | Dimensions            |
| ----------------------------- | --------- | --------------------- |
| `/train/images`               | Byte      | 70478 x 3 x 28 x 28   |
| `/train/codes`                | Char      | 70478 x 32            |
